# DendAttn 架构详解

> 基于论文 "Biologically Inspired Neuron Structures for Versatile Foundation Models: Augmenting Memory Capacity, Temporal Dynamics, and Information Segregation via Native Dendritic Mechanisms"

---

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DendAttn 整体架构                                   │
│                   (受生物树突神经元启发的线性注意力机制)                        │
└─────────────────────────────────────────────────────────────────────────┘

输入: hidden_states [batch, seq_len, hidden_size]
   │
   ├──────────────┬──────────────┬──────────────┐
   ↓              ↓              ↓              ↓
Q投影          K投影          V投影      β/γ门控参数
   │              │              │              │
   └──────────────┴──────────────┴──────────────┘
                    ↓
   ┌────────────────────────────────────────────┐
   │   多头拆分 (num_heads个头)                   │
   └────────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────────┐
   │  每个头的多分支扩展 (ratio个分支)               │
   │  - shared_head: 共享分支 (始终激活)           │
   │  - routing branches: 路由分支 (稀疏激活)      │
   └────────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────────┐
   │  稀疏路由机制 (Router)                        │
   │  - 通过gate网络计算路由权重                     │
   │  - TopK选择激活的分支                         │
   └────────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────────┐
   │  短卷积层 (Short Convolution)                │
   │  - 对Q/K/V分别进行1D卷积                      │
   └────────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────────┐
   │  Block-Sparse门控更新                        │
   │  - Q/K分块重叠处理 (re_process)               │
   │  - 局部密集+全局稀疏的连接模式                  │
   └────────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────────┐
   │  Gated Delta Rule 计算                      │
   │  S_t = α_t * S_{t-1} + K^T * V             │
   │  O_t = Q_t * S_t                           │
   └────────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────────┐
   │  路由权重加权聚合                              │
   │  O = Σ router_weight * O_branch            │
   └────────────────────────────────────────────┘
                    ↓
   ┌────────────────────────────────────────────┐
   │  输出门控归一化 + 输出投影                      │
   └────────────────────────────────────────────┘
                    ↓
输出: output [batch, seq_len, hidden_size]
```

---

## 二、三个模块的演进关系

```
演进路径：
GatedDeltaNetp  →  MobGatedDeltaNet  →  MobGatedDeltaNetMoE
    (基线)           (多分支)              (多分支+MoE路由)
     │                  │                      │
     │                  │                      │
     ↓                  ↓                      ↓
单一分支           多分支扩展              多分支+稀疏激活
线性注意力          但全部激活              (完整DendAttn)
```

| 模块 | 文件位置 | 特点 |
|------|----------|------|
| GatedDeltaNetp | `gated_deltanet_p/layer.py` | 单分支基线 |
| MobGatedDeltaNet | `mob_gated_deltanet/layer.py` | 多分支密集激活 |
| MobGatedDeltaNetMoE | `mob_gated_deltanet_moe/layer.py` | 多分支稀疏激活（完整版） |

---

## 三、配置参数说明

```python
# 示例配置
batch_size (B) = 2
seq_len (L) = 1024
hidden_size (D) = 2048
num_heads (H) = 8
head_dim (d) = 256          # key_dim = H * d = 2048
expand_v = 2
head_v_dim = 512            # value_dim = H * head_v_dim = 4096
ratio (E) = 8               # 每个头的分支数
shared_head (Es) = 1        # 共享分支数
topk (K) = 2                # 每次激活的路由分支数
num_block (N) = 2           # Q/K分块数
overlap (O) = 64            # 块间重叠大小
```

---

## 四、完整前向传播流程（带维度标注）

### 阶段0: 输入
```
hidden_states: [B=2, L=1024, D=2048]
```

### 阶段1: Q/K/V投影
```
self.q_proj: Linear(2048 → 2048, bias=False)
    参数矩阵: [2048, 2048] = 4M 参数
self.k_proj: Linear(2048 → 2048, bias=False)
    参数矩阵: [2048, 2048] = 4M 参数
self.v_proj: Linear(2048 → 4096, bias=False)
    参数矩阵: [2048, 4096] = 8M 参数

q = self.q_proj(hidden_states)
    输出: [2, 1024, 2048]
k = self.k_proj(hidden_states)
    输出: [2, 1024, 2048]
v = self.v_proj(hidden_states)
    输出: [2, 1024, 4096]
```

### 阶段2: 重排成多头形式
```
q = rearrange(q, 'b t (h d) -> h b t d', h=8)
    [2, 1024, 2048] → [H=8, B=2, L=1024, d=256]

k = rearrange(k, 'b t (h d) -> h b t d', h=8)
    [2, 1024, 2048] → [8, 2, 1024, 256]

v = rearrange(v, 'b t (h d) -> h b t d', h=8)
    [2, 1024, 4096] → [8, 2, 1024, 512]
```

### 阶段3: 稀疏路由计算 (sparse函数)

#### 3.1 重排Q用于路由计算
```
q: [8, 2, 1024, 256]
→ rearrange('h b l d -> (h b) l d')
→ [HB=16, L=1024, d=256]
```

#### 3.2 计算路由分数
```
self.gate: Linear(256 → 7, bias=False)  # ratio - shared_head
    参数矩阵: [256, 7] = 1.8K 参数/头 × 8头

router_logits = self.gate(q)
    输入: [16, 1024, 256]
    输出: [16, 1024, 7]  # 对7个routing分支打分
```

#### 3.3 TopK选择
```
scores = softmax(router_logits, dim=2)
    [16, 1024, 7]
routing_weights, selected_memories = topk(scores, k=2)
    routing_weights: [16, 1024, 2]
    selected_memories: [16, 1024, 2]  # 选中的分支索引
```

#### 3.4 构建完整路由权重
```
router_weight_full: [16, 1024, 8]  初始化为0
scatter操作: 将topk权重填入对应位置
shared_head部分: [:, :, 0:1] = 1/1 = 1.0
归一化后每个token在每个头的权重和 = 1.0

→ rearrange('(h b) l e -> e b l h', h=8)
router_weight_full: [E=8, B=2, L=1024, H=8]
router_mask: [8, 2, 1024, 8] (bool转int)
```

### 阶段4: 多分支扩展

#### 4.1 专家投影定义
```
对每个头h，应用专家投影:
self.k_proj_expand[h]: Linear(256 → 256*8=2048, bias=False)
    参数矩阵: [256, 2048] × 8个头 = 512K × 8 = 4M 参数

self.q_proj_expand[h]: Linear(256 → 2048, bias=False)
    参数矩阵: [256, 2048] × 8个头 = 4M 参数
```

#### 4.2 K的扩展
```
k初始: [8, 2, 1024, 256]
对每个头i应用k_proj_expand[i]:
    k[i]: [2, 1024, 256] → [2, 1024, 2048]
stack后: [8, 2, 1024, 2048]
```

#### 4.3 Q的扩展
```
q初始: [8, 2, 1024, 256]
类似操作后: [8, 2, 1024, 2048]
```

#### 4.4 重排成分支维度
```
k = rearrange(k, 'h b l (e d) -> e h b l d', e=8)
    [8, 2, 1024, 2048] → [E=8, H=8, B=2, L=1024, d=256]

q同样: [8, 8, 2, 1024, 256]
```

#### 4.5 V的复制（每个分支共享同一个V）
```
v初始: [8, 2, 1024, 512]
v = repeat(v, 'h b l d -> e h b l d', e=8)
    → [8, 8, 2, 1024, 512]
```

#### 4.6 合并以便后续处理
```
k = rearrange(k, 'e h b l d -> (e b) l (h d)')
    [8, 8, 2, 1024, 256] → [EB=16, L=1024, HD=2048]
q同样: [16, 1024, 2048]
v: [16, 1024, 4096]
```

### 阶段5: 短卷积处理
```
self.q_conv1d: ShortConvolution(hidden_size=2048, kernel_size=4)
    卷积核: [2048, 1, 4]

q = self.q_conv1d(q)
    输入: [16, 1024, 2048]
    输出: [16, 1024, 2048]  # 形状不变，但加入局部依赖

k = self.k_conv1d(k)
    [16, 1024, 2048] → [16, 1024, 2048]

v = self.v_conv1d(v)
    [16, 1024, 4096] → [16, 1024, 4096]
```

#### 5.1 应用路由掩码
```
重排回分支形式:
k = rearrange(k, '(e b) l (h d) -> e b l h d', e=8, h=8)
    [16, 1024, 2048] → [8, 2, 1024, 8, 256]

应用掩码:
k = k * router_mask[..., None]
    [8, 2, 1024, 8, 256] * [8, 2, 1024, 8, 1]
    → [8, 2, 1024, 8, 256]  # 未选中的分支被置0

q, v同样操作
```

### 阶段6: 门控参数计算
```
self.b_proj: Linear(2048 → 64, bias=False)  # H*E = 8*8
    参数矩阵: [2048, 64] = 128K 参数

self.a_proj: Linear(2048 → 64, bias=False)
    参数矩阵: [2048, 64] = 128K 参数

beta = self.b_proj(hidden_states).sigmoid()
    [2, 1024, 2048] → [2, 1024, 64]
    → rearrange('b l (e h) -> e b l h', e=8, h=8)
    → [8, 2, 1024, 8]
    → 乘以router_mask: [8, 2, 1024, 8]

g = -A_log.exp() * softplus(a_proj(hidden_states) + dt_bias)
    计算流程同上，输出: [8, 2, 1024, 8]
```

### 阶段7: Block-Sparse处理 (re_process)

#### 7.1 重排为统一格式
```
q = rearrange(q, 'e b l h d -> b l (e h) d', e=8)
    [8, 2, 1024, 8, 256] → [2, 1024, 64, 256]
k, v, g, beta同样操作
```

#### 7.2 Q/K分块（re_process函数）
```python
def re_process(x: torch.Tensor, n: int, overlap: int) -> torch.Tensor:
    """
    将Q/K的最后维度分割成n个重叠的窗口
    对应论文中的Block-Sparse Gating机制
    """
    b, l, h, d = x.shape
    window_size = (d + (n - 1) * overlap) // n
    step = window_size - overlap
    slices = [x[..., i*step : i*step + window_size] for i in range(n)]
    return torch.cat(slices, dim=2)
```

```
参数: n=2 (num_block), overlap=64

对于Q: [2, 1024, 64, 256]
    window_size = (256 + (2-1)*64) // 2 = 320/2 = 160
    step = 160 - 64 = 96

    切片0: [..., 0:160]     → [2, 1024, 64, 160]
    切片1: [..., 96:256]    → [2, 1024, 64, 160]

    cat(dim=2) → [2, 1024, 128, 160]
    说明: 64个头变成128个头，每个头维度从256降到160

K同样: [2, 1024, 64, 256] → [2, 1024, 128, 160]
```

#### 7.3 V/g/beta复制（每块共享）
```
v = repeat(v, 'b t h d -> b t (k h) d', k=2)
    [2, 1024, 64, 512] → [2, 1024, 128, 512]

g: [2, 1024, 64] → [2, 1024, 128]
beta: [2, 1024, 64] → [2, 1024, 128]
```

### 阶段8: Gated Delta Rule核心计算
```
o, recurrent_state = chunk_gated_delta_rule(
    q=[2, 1024, 128, 160],
    k=[2, 1024, 128, 160],
    v=[2, 1024, 128, 512],
    g=[2, 1024, 128],
    beta=[2, 1024, 128]
)

内部计算（简化）:
    对每个头h:
        S_t = g_t * S_{t-1} + beta_t * k_t^T @ v_t
            S形状: [B, H, d_k, d_v] = [2, 128, 160, 512]
        o_t = q_t @ S_t
            [2, 1, 160] @ [2, 160, 512] → [2, 1, 512]

输出: o = [2, 1024, 128, 512]
```

#### 8.1 跨块聚合
```
o = rearrange(o, 'b t (k h) d -> b t h k d', k=2)
    [2, 1024, 128, 512] → [2, 1024, 64, 2, 512]

o = o.sum(-2)  # 将2个块的结果相加
    → [2, 1024, 64, 512]
```

### 阶段9: 多分支加权聚合

#### 9.1 重排回分支维度
```
o = rearrange(o, 'b l (e h) d -> e b l h d', e=8)
    [2, 1024, 64, 512] → [8, 2, 1024, 8, 512]
```

#### 9.2 使用路由权重加权求和
```
router_weight_full: [8, 2, 1024, 8]
o = einsum('eblhd,eblh->blhd', o, router_weight_full)
    将8个分支按权重聚合
    → [2, 1024, 8, 512]
```

### 阶段10: 输出门控与投影

#### 10.1 计算输出门控
```
self.g_proj: Linear(2048 → 4096, bias=False)
    参数矩阵: [2048, 4096] = 8M 参数

g = self.g_proj(hidden_states)
    [2, 1024, 2048] → [2, 1024, 4096]
g = rearrange(g, 'b l (h d) -> b l h d', h=8)
    → [2, 1024, 8, 512]
```

#### 10.2 RMSNorm + Gate
```
self.o_norm: FusedRMSNormSwishGate(512)
o = self.o_norm(o, g)
    输入o: [2, 1024, 8, 512]
    输入g: [2, 1024, 8, 512]
    输出: [2, 1024, 8, 512]
```

#### 10.3 最终投影
```
o = rearrange(o, 'b t h d -> b t (h d)')
    [2, 1024, 8, 512] → [2, 1024, 4096]

self.o_proj: Linear(4096 → 2048, bias=False)
    参数矩阵: [4096, 2048] = 8M 参数

o = self.o_proj(o)
    [2, 1024, 4096] → [2, 1024, 2048]
```

### 最终输出
```
output: [B=2, L=1024, D=2048]
```

---

## 五、关键矩阵参数量统计

| 模块 | 形状 | 参数量 |
|------|------|--------|
| q_proj | [2048, 2048] | 4.19M |
| k_proj | [2048, 2048] | 4.19M |
| v_proj | [2048, 4096] | 8.39M |
| q_proj_expand (×8头) | [256, 2048] ×8 | 4.19M |
| k_proj_expand (×8头) | [256, 2048] ×8 | 4.19M |
| gate (×8头) | [256, 7] ×8 | 0.01M |
| a_proj | [2048, 64] | 0.13M |
| b_proj | [2048, 64] | 0.13M |
| g_proj | [2048, 4096] | 8.39M |
| o_proj | [4096, 2048] | 8.39M |
| q/k/v_conv1d | 卷积核 | ~0.05M |
| **总计** | | **~42M** |

> 注: 单层参数量约42M，24层模型总参数量约1B

---

## 六、内存占用分析（前向传播）

| 张量名称 | 形状 | 内存(MB) FP16 |
|----------|------|---------------|
| hidden_states | [2, 1024, 2048] | 8.4 |
| q/k (各) | [2, 1024, 2048] | 8.4 ×2 |
| v | [2, 1024, 4096] | 16.8 |
| q_expanded | [8, 8, 2, 1024, 256] | 67.1 |
| k_expanded | [8, 8, 2, 1024, 256] | 67.1 |
| v_expanded | [8, 8, 2, 1024, 512] | 134.2 |
| router_weight_full | [8, 2, 1024, 8] | 0.26 |
| recurrent_state (峰值) | [2, 128, 160, 512] | 26.2 |
| output | [2, 1024, 8, 512] | 16.8 |
| **峰值总内存 (单层)** | | **~350MB** |

### 与Transformer的KV Cache对比

```
DendAttn:    固定 26.2MB (recurrent_state，不随seq_len增长)
Transformer: 16.8MB × seq_len / 1024 (线性增长)

当seq_len=64k时:
  DendAttn:    26.2MB (不变)
  Transformer: 1075MB (64倍增长)
```

---

## 七、稀疏性分析

### 路由激活模式（每个token在每个头）
```
总分支数: 8
共享分支: 1 (始终激活)
路由分支: 从7个候选中选2个

实际激活: 1 + 2 = 3个分支
激活率: 3/8 = 37.5%
计算量降低: 1 - 0.375 = 62.5%
```

### 全序列的激活分布
```
batch=2, seq_len=1024, num_heads=8

理论最大激活数: 2 × 1024 × 8 × 8 = 131,072个分支
实际激活数: 2 × 1024 × 8 × 3 = 49,152个分支

稀疏度: 49152/131072 = 37.5%
```

---

## 八、与论文的对应关系

| 论文概念 | 代码实现 | 文件位置 |
|---------|---------|----------|
| 单分支树突 | GatedDeltaNetp | `gated_deltanet_p/layer.py` |
| 多分支树突 | ratio参数 + k_proj_expand | 第173-174行 |
| 共享分支 | shared_head参数 | 第137行 |
| 路由分支 | topk参数 + sparse函数 | 第157行, 241-273行 |
| Block-Sparse门控 | re_process + num_block | 第36-73行, 381-385行 |
| 信息隔离 | router_mask掩码 | 第347-350行 |
| 状态更新 | chunk_gated_delta_rule | 第390-401行 |

---

## 九、核心公式

### Delta Rule状态更新
$$S_t = \alpha_t \cdot S_{t-1} + K_t^T \cdot V_t$$

$$O_t = Q_t \cdot S_t$$

### 门控Delta Rule
$$S_t = g_t \cdot S_{t-1} + \beta_t \cdot K_t^T \cdot V_t$$

其中:
- $g_t = -\exp(A_{log}) \cdot \text{softplus}(a\_proj(x_t) + dt\_bias)$：门控衰减因子
- $\beta_t = \sigma(b\_proj(x_t))$：输入门控

### 多分支聚合
$$O = \sum_{e=1}^{E} w_e \cdot O_e$$

其中 $w_e$ 为路由权重，满足 $\sum_e w_e = 1$

---

## 十、生物学对应

| 生物学概念 | 模型对应 | 功能作用 |
|-----------|----------|----------|
| 树突分支 | ratio个专家分支 | 扩展记忆容量 |
| 树突隔室 | num_block个块 | 局部信息处理 |
| 轴突电阻 | overlap参数 | 相邻隔室间通信 |
| 膜电位 | recurrent_state | 记忆状态 |
| 稀疏激活 | TopK路由 | 信息隔离 |
| 突触权重 | k_proj_expand/q_proj_expand | 输入调制 |
| 输出整合 | 加权聚合 | 多分支信号汇聚 |

---

## 十一、性能优势总结

1. **推理加速**: 33.7× (512k序列长度)
2. **计算复杂度**: O(n) vs Transformer的O(n²)
3. **内存效率**: 固定状态空间 vs 线性增长的KV Cache
4. **长序列外推**: 训练2k，可外推到128k
5. **多任务性能**: 提升14.27%
