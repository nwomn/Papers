# TileLang 与 NVIDIA cuTile 对比

## 1. 背景

### 1.1 TileLang（字节跳动）

- **开源时间**：2025 年初
- **开发者**：字节跳动
- **GitHub**：[tile-ai/tilelang](https://github.com/tile-ai/tilelang)
- **论文**：arXiv:2504.17577

### 1.2 cuTile（NVIDIA）

- **发布时间**：CUDA 13.1（2025 年）
- **开发者**：NVIDIA
- **GitHub**：[NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python)
- **文档**：[docs.nvidia.com/cuda/cutile-python](https://docs.nvidia.com/cuda/cutile-python/)

---

## 2. 核心对比

| 特性 | TileLang | cuTile |
|------|----------|--------|
| **开发者** | 字节跳动 | NVIDIA |
| **编程语言** | Python | Python（C++ 计划中） |
| **底层实现** | 基于 TVM/MLIR | 基于 Tile IR (MLIR) |
| **硬件支持** | NVIDIA, AMD, Apple, WebGPU | NVIDIA（仅 Blackwell 及以上） |
| **开源协议** | Apache 2.0 | Apache 2.0 |
| **成熟度** | 相对成熟，已在生产中使用 | 较新，2025 年发布 |

---

## 3. 硬件支持对比

### 3.1 TileLang

| 厂商 | 支持设备 | 特殊优化 |
|------|---------|----------|
| **NVIDIA** | H100, A100, V100, RTX 4090/3090 | Auto TMA, WGMMA |
| **AMD** | MI250, MI300X | Auto MatrixCore |
| **Apple** | Metal 设备 | Metal Shader |
| **Web** | WebGPU | WGSL Codegen |

### 3.2 cuTile

| 要求 | 详情 |
|------|------|
| **GPU** | Compute Capability 10.x 或 12.x（Blackwell 及以上） |
| **驱动** | NVIDIA Driver R580+ |
| **CUDA** | CUDA Toolkit 13.1+ |

**关键限制**：cuTile 目前**不支持** Ampere (A100) 和 Hopper (H100)，仅支持 Blackwell 架构。

---

## 4. 编程模型对比

### 4.1 TileLang 示例

```python
import tilelang as tl

@tl.kernel
def matmul(
    A: tl.Tensor[M, K, tl.float16],
    B: tl.Tensor[K, N, tl.float16],
    C: tl.Tensor[M, N, tl.float32],
):
    tile_m, tile_n, tile_k = 128, 128, 32

    for m, n in tl.grid(M // tile_m, N // tile_n):
        acc = tl.zeros([tile_m, tile_n], tl.float32)

        for k in range(K // tile_k):
            a_tile = A[m*tile_m:(m+1)*tile_m, k*tile_k:(k+1)*tile_k]
            b_tile = B[k*tile_k:(k+1)*tile_k, n*tile_n:(n+1)*tile_n]
            acc += a_tile @ b_tile

        C[m*tile_m:(m+1)*tile_m, n*tile_n:(n+1)*tile_n] = acc
```

### 4.2 cuTile 示例

```python
import cutile as ct

@ct.kernel
def matmul(
    A: ct.Tensor[M, K, ct.float16],
    B: ct.Tensor[K, N, ct.float16],
    C: ct.Tensor[M, N, ct.float32],
):
    # cuTile 自动处理 block-level 并行和异步
    tile_m, tile_n, tile_k = 128, 128, 32

    acc = ct.zeros([tile_m, tile_n], ct.float32)

    for k in ct.range(K // tile_k):
        a_tile = ct.load(A, tile_m, tile_k)
        b_tile = ct.load(B, tile_k, tile_n)
        acc = ct.mma(a_tile, b_tile, acc)  # 自动使用 Tensor Core

    ct.store(C, acc)
```

**对比**：
- TileLang 需要显式定义 grid 循环
- cuTile 自动处理 block-level 并行
- 两者都自动使用 Tensor Core

---

## 5. 技术架构对比

### 5.1 编译流程

**TileLang**：
```
Python DSL → TVM IR → LLVM IR → PTX/SASS (NVIDIA)
                    → ROCm (AMD)
                    → Metal (Apple)
```

**cuTile**：
```
Python DSL → Tile IR (MLIR) → LLVM IR → PTX/SASS
                            ↑
                     CUDA Tile IR Spec
```

### 5.2 核心技术

| 技术 | TileLang | cuTile |
|------|----------|--------|
| **IR 基础** | TVM + MLIR | CUDA Tile IR (MLIR) |
| **Tensor Core 抽象** | 自动检测 | 自动，原生支持 |
| **内存管理** | 半自动 | 自动（TMA） |
| **流水线调度** | 用户注解 | 自动 |

---

## 6. 关系分析

### 6.1 是否有联系？

**没有直接联系**，但有**相似的设计理念**：

| 共同点 | 说明 |
|--------|------|
| **Tile 抽象** | 都以 Tile（瓦片）作为基本计算单元 |
| **Python 优先** | 都提供 Python DSL |
| **自动优化** | 都自动利用硬件特性（Tensor Core、TMA 等） |
| **MLIR 技术栈** | 都基于 MLIR 生态 |

### 6.2 主要区别

| 区别 | TileLang | cuTile |
|------|----------|--------|
| **定位** | 跨平台通用 DSL | NVIDIA 专用 |
| **生态** | 独立项目 | CUDA 官方组件 |
| **硬件绑定** | 松耦合 | 紧耦合（最新架构） |
| **控制粒度** | 更多用户控制 | 更多自动优化 |

---

## 7. mHC 论文中使用 TileLang 的原因

论文提到：

> "We efficiently implement the majority of kernels (excluding Eq. (14) to (15)) using TileLang."

### 7.1 选择 TileLang 的可能原因

1. **时间线**：mHC 论文（2025.12）之前，cuTile 可能尚未发布或不够成熟
2. **硬件兼容**：实验使用 H100/A100，cuTile 不支持
3. **跨平台需求**：可能需要在不同硬件上测试
4. **已有经验**：DeepSeek 团队可能已熟悉 TileLang

### 7.2 如果现在重写

如果目标硬件是 **Blackwell (B100/B200)**，cuTile 可能是更好的选择：
- 原生 NVIDIA 支持
- 更好的 TMA 和 WGMMA 集成
- 官方维护和优化

---

## 8. 实践建议

### 8.1 何时用 TileLang

- 需要**跨平台**（NVIDIA + AMD + Apple）
- 目标硬件是 **Ampere/Hopper**（A100/H100）
- 需要**更多控制**（自定义调度策略）
- 已有 TVM/TileLang 经验

### 8.2 何时用 cuTile

- 目标硬件是 **Blackwell** 及以上
- 追求**最大 NVIDIA 性能**
- 希望利用**最新 CUDA 特性**（TMA、异步拷贝等）
- 不需要跨平台

### 8.3 混合使用

```python
# 伪代码：根据硬件选择后端
if gpu_arch >= "sm_100":  # Blackwell
    use_cutile()
else:
    use_tilelang()
```

---

## 9. 总结

| 问题 | 答案 |
|------|------|
| **TileLang 和 cuTile 有联系吗？** | 无直接联系，但设计理念相似 |
| **主要区别？** | TileLang 跨平台，cuTile 仅限 NVIDIA Blackwell+ |
| **哪个更好？** | 取决于硬件和需求 |
| **mHC 为什么用 TileLang？** | 兼容 H100/A100，且论文时 cuTile 可能不可用 |

**一句话总结**：

TileLang 是跨平台的通用方案，cuTile 是 NVIDIA 的官方最新硬件专用方案——选择取决于你的目标硬件。

---

## 10. 参考链接

- [NVIDIA CUDA Tile 官方页面](https://developer.nvidia.com/cuda/tile)
- [cuTile Python 文档](https://docs.nvidia.com/cuda/cutile-python/)
- [NVIDIA 博客：Simplify GPU Programming with CUDA Tile](https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/)
- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [TileLang 论文](https://arxiv.org/abs/2504.17577)
