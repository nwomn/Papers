# TileLang 框架详解

## 1. 什么是 TileLang？

**TileLang** 是一个用于高性能 AI 内核开发的**领域特定语言 (DSL)**，由字节跳动开源。

**核心定位**：在 Python 中编写高性能 GPU 内核，无需手写 CUDA。

**GitHub**: [tile-ai/tilelang](https://github.com/tile-ai/tilelang)
**论文**: [arXiv:2504.17577](https://arxiv.org/abs/2504.17577)

---

## 2. 为什么需要 TileLang？

### 2.1 传统 CUDA 开发的痛点

| 问题 | 描述 |
|------|------|
| **开发难度高** | 需要精通 CUDA、线程模型、内存层次 |
| **代码冗长** | 大量样板代码（thread binding、shared memory 管理） |
| **调试困难** | GPU 调试工具有限，bug 难以定位 |
| **可移植性差** | NVIDIA/AMD/Apple 需要不同实现 |
| **优化繁琐** | 手动优化 tiling、向量化、流水线 |

### 2.2 现有方案对比

| 方案 | 抽象层次 | 性能 | 开发效率 | 可移植性 |
|------|---------|------|----------|----------|
| **Raw CUDA** | 最低 | 最高（手动优化） | 低 | 差 |
| **cuBLAS/cuDNN** | 高 | 高 | 高 | 中 |
| **Triton** | 中 | 高 | 中 | 中 |
| **TileLang** | 中-高 | 高 | **高** | **好** |

---

## 3. TileLang 的核心设计理念

### 3.1 三大设计原则

| 原则 | 含义 |
|------|------|
| **Pythonic** | 完全在 Python 中编程，与 PyTorch 生态无缝集成 |
| **Dataflow-Centric** | 以数据流为中心，自动处理调度细节 |
| **Annotation-Based** | 通过注解定制优化，而非手写底层代码 |

### 3.2 核心抽象：Tile（瓦片）

**Tile** = 数据的基本处理单元

```
GPU 计算模型:
┌─────────────────────────────────────────┐
│              Global Memory               │
└─────────────────────────────────────────┘
                    ↓ 分块加载
┌─────────────────────────────────────────┐
│    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│    │Tile │ │Tile │ │Tile │ │Tile │     │  ← Shared Memory
│    │ 0,0 │ │ 0,1 │ │ 1,0 │ │ 1,1 │     │
│    └─────┘ └─────┘ └─────┘ └─────┘     │
└─────────────────────────────────────────┘
                    ↓ 并行计算
┌─────────────────────────────────────────┐
│   Thread Block 0  │  Thread Block 1     │  ← 多个 SM 并行
└─────────────────────────────────────────┘
```

TileLang 自动处理：
- Tile 大小选择
- 内存加载/存储
- Thread 分配
- 流水线调度

---

## 4. TileLang vs 其他方案

### 4.1 与 Triton 对比

| 特性 | Triton | TileLang |
|------|--------|----------|
| 开发者 | OpenAI | 字节跳动 |
| 编程模型 | Block-based | Tile-based |
| 优化控制 | 手动 + 编译器 | **注解驱动** |
| AMD 支持 | 部分 | **完整** |
| Tensor Core | 手动 | **自动** |

### 4.2 代码对比示例

**任务**：矩阵乘法 C = A @ B

**Triton 实现**（约 50 行）：
```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)
```

**TileLang 实现**（约 20 行）：
```python
import tilelang as tl

@tl.kernel
def matmul(
    A: tl.Tensor[M, K, tl.float16],
    B: tl.Tensor[K, N, tl.float16],
    C: tl.Tensor[M, N, tl.float32],
):
    # 声明 tile 大小
    tile_m, tile_n, tile_k = 128, 128, 32

    # 自动处理分块和循环
    for m, n in tl.grid(M // tile_m, N // tile_n):
        acc = tl.zeros([tile_m, tile_n], tl.float32)

        for k in range(K // tile_k):
            a_tile = A[m*tile_m:(m+1)*tile_m, k*tile_k:(k+1)*tile_k]
            b_tile = B[k*tile_k:(k+1)*tile_k, n*tile_n:(n+1)*tile_n]
            acc += a_tile @ b_tile  # 自动使用 Tensor Core

        C[m*tile_m:(m+1)*tile_m, n*tile_n:(n+1)*tile_n] = acc
```

**优势**：
- 代码量减少 60%
- 无需手动管理 pointer、stride
- 自动使用 Tensor Core
- 自动内存优化

---

## 5. TileLang 在 mHC 中的应用

### 5.1 mHC 需要的自定义 Kernel

| Kernel | 功能 | 挑战 |
|--------|------|------|
| **Kernel 1** | 计算 $\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}, \mathcal{H}^{\text{res}}$ | RMSNorm + 矩阵乘 + 混合精度 |
| **Kernel 2** | Sigmoid + 缩放 | 轻量操作，需融合以减少 kernel launch |
| **Kernel 3** | Sinkhorn-Knopp 迭代 | 20 次迭代，需片上计算 |
| **Kernel 4** | $\mathcal{F}_{\text{pre}} = \mathcal{H}^{\text{pre}} x_l$ | 小矩阵乘 |
| **Kernel 5** | $\mathcal{F}_{\text{post,res}}$ | 融合 post 和 res 映射 |

### 5.2 为什么选择 TileLang？

| 需求 | TileLang 的支持 |
|------|-----------------|
| **Kernel Fusion** | 注解驱动的自动融合 |
| **混合精度** | 原生支持 float32/float16/bfloat16 |
| **Sinkhorn 迭代** | 可在单个 kernel 内实现循环 |
| **H100 优化** | 自动 TMA/WGMMA |
| **快速迭代** | Python 开发，无需编译 CUDA |

### 5.3 示例：Sinkhorn-Knopp Kernel

```python
import tilelang as tl

@tl.kernel
def sinkhorn_knopp_kernel(
    H_res: tl.Tensor[batch, n, n, tl.float32],  # 输入矩阵
    output: tl.Tensor[batch, n, n, tl.float32],  # 输出矩阵
    num_iters: int = 20,
):
    """
    将 H_res 投影到双随机矩阵流形
    """
    # 每个 batch 独立处理
    for b in tl.grid(batch):
        # 加载到 shared memory
        M = tl.exp(H_res[b])  # [n, n]

        # Sinkhorn-Knopp 迭代（片上计算）
        for _ in range(num_iters):
            # 行归一化
            row_sum = tl.sum(M, axis=1, keepdims=True)  # [n, 1]
            M = M / row_sum

            # 列归一化
            col_sum = tl.sum(M, axis=0, keepdims=True)  # [1, n]
            M = M / col_sum

        # 写回 global memory
        output[b] = M
```

**关键优化**：
1. **片上迭代**：20 次迭代全部在 shared memory 完成，避免反复读写 global memory
2. **自动向量化**：TileLang 自动将 sum/div 向量化
3. **混合精度**：可指定中间计算用 float32，输入输出用 float16

---

## 6. TileLang 支持的硬件

| 厂商 | 支持的设备 | 特殊优化 |
|------|-----------|----------|
| **NVIDIA** | H100, A100, V100, RTX 4090/3090 | Auto TMA, WGMMA |
| **AMD** | MI250, MI300X | Auto MatrixCore |
| **Apple** | Metal 设备 | Metal Shader |
| **Web** | WebGPU | WGSL Codegen |

---

## 7. 总结

### 7.1 "使用 TileLang 简化开发"的含义

| 传统方式 | TileLang 方式 |
|----------|---------------|
| 手写 CUDA kernel | Python 代码 + 注解 |
| 手动管理 shared memory | 自动优化内存层次 |
| 手动 thread binding | 自动并行化 |
| 手动 Tensor Core 调用 | 自动检测和使用 |
| 不同硬件需要不同代码 | 一份代码多平台运行 |

### 7.2 mHC 论文中的具体收益

> "We efficiently implement the majority of kernels (excluding Eq. (14) to (15)) using TileLang. This framework streamlines the implementation of kernels with complex calculation process and allows us to fully utilize the memory bandwidth with minimal engineering effort."

**收益**：
1. 复杂计算流程（Sinkhorn 迭代）的快速实现
2. 充分利用内存带宽
3. 最小化工程投入

---

## 8. 学习资源

- **GitHub**: [tile-ai/tilelang](https://github.com/tile-ai/tilelang)
- **论文**: [TileLang: A Composable Tiled Programming Model](https://arxiv.org/abs/2504.17577)
- **NVIDIA CUDA Tile**（相关技术）: [NVIDIA Developer Blog](https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/)
