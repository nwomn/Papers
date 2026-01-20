# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

## 基本信息
- **作者**: DeepSeek-AI & 北京大学
- **代码**: https://github.com/deepseek-ai/Engram
- **核心贡献**: 提出 Engram 模块，将条件记忆作为 MoE 的互补稀疏轴

---

## 论文结构与页号索引

### 1. Introduction (p.1-2)
**核心问题**:
- 语言建模包含两类子任务：组合推理（需要深度动态计算）和知识检索（局部、静态、高度模式化）
- 当前 Transformer 缺乏原生知识查找原语，被迫通过计算模拟检索
- 例如：识别"Diana, Princess of Wales"需要消耗多层 Attention+FFN，本质是运行时重建静态查找表

**解决方案**:
- 提出**条件记忆（Conditional Memory）**作为新的稀疏轴
- 条件计算（MoE）：稀疏激活参数处理动态逻辑
- 条件记忆（Engram）：稀疏查找操作检索静态知识

**主要发现**:
- U 形稀疏分配定律：~20-25% 参数分配给 Engram 最优
- Engram-27B 在等参数等FLOPs下优于 MoE-27B
- 不仅知识任务提升（MMLU +3.4），推理任务提升更大（BBH +5.0）
- 长上下文显著改善（Multi-Query NIAH: 84.2→97.0）

---

### 2. Architecture (p.3-6)

#### 2.1 Overview (p.3)
- Engram 是条件记忆模块，结构性分离静态模式存储与动态计算
- 两个功能阶段：**检索（Retrieval）** 和 **融合（Fusion）**

#### 2.2 Sparse Retrieval via Hashed N-grams (p.3-4)
**Tokenizer 压缩**:
- 将语义等价 token 映射到同一 ID（如 `Apple` vs `␣apple`）
- 使用 NFKC 标准化 + 小写化
- 128k tokenizer 实现 23% 词表压缩（见附录 C, p.33）

**多头哈希**:
- 每个 N-gram 阶数 n 使用 K 个哈希头
- 确定性函数 φ_{n,k} 映射到素数大小的嵌入表
- 公式 (1): z_{t,n,k} = φ_{n,k}(g_{t,n}), e_{t,n,k} = E_{n,k}[z_{t,n,k}]
- 公式 (2): 拼接所有检索嵌入得到最终记忆向量 e_t

#### 2.3 Context-aware Gating (p.4)
- 检索嵌入是上下文无关的先验，可能有哈希碰撞或多义性噪声
- 使用当前隐藏状态 h_t 作为动态 Query，检索记忆 e_t 作为 Key/Value
- 公式 (3): k_t = W_K·e_t, v_t = W_V·e_t
- 公式 (4): α_t = σ(RMSNorm(h_t)^T·RMSNorm(k_t)/√d) ∈ (0,1)
- 门控输出: ṽ_t = α_t · v_t（语义不匹配时门控趋近于零）
- 公式 (5): 加入深度因果卷积（kernel=4, dilation=max N-gram order）+ SiLU 激活

#### 2.4 Integration with Multi-branch Architecture (p.5)
- 默认使用 mHC (Manifold-Constrained Hyper-Connections, M=4) 作为骨干
- 参数共享策略：单一嵌入表和 W_V 跨所有分支共享，M 个独立 W_K^(m) 实现分支特定门控
- 公式 (6): 分支特定门控信号计算
- 线性投影可融合为单个 FP8 矩阵乘法

#### 2.5 System Efficiency (p.6)
**训练阶段**:
- 模型并行：嵌入表跨 GPU 分片
- All-to-All 通信收集激活行和分发梯度

**推理阶段**:
- 确定性检索支持**预取-重叠**策略
- 索引在前向传播前已知，可异步从主机内存通过 PCIe 检索
- Engram 放置在特定层，利用前序层计算作为缓冲掩盖通信延迟
- **放置权衡**: 早放可卸载局部模式重建，但门控精度较低；晚放有更好上下文但错过早期干预

**多级缓存层次**:
- N-gram 服从 Zipf 分布，高频嵌入缓存在快速存储（GPU HBM / Host DRAM）
- 长尾稀有模式存储在高容量慢速介质（NVMe SSD）

---

### 3. Scaling Laws and Sparsity Allocation (p.7-8)

#### 3.1 Optimal Allocation Ratio (p.7)
**参数定义**:
- P_tot: 总可训练参数（不含词表嵌入和 LM head）
- P_act: 每 token 激活参数（决定 FLOPs）
- P_sparse = P_tot - P_act: 非激活参数（"免费"扩展预算）

**分配比 ρ ∈ [0,1]**:
- 公式 (7): P_MoE^(sparse) = ρ·P_sparse, P_Engram = (1-ρ)·P_sparse
- ρ=1: 纯 MoE；ρ<1: 减少专家数，释放参数给 Engram

**实验设置**:
- C = 2×10^20 FLOPs: P_tot≈5.7B, P_act=568M, 106 专家
- C = 6×10^20 FLOPs: P_tot≈9.9B, P_act=993M, 99 专家

**结果 (Figure 3 左, p.6)**:
- U 形曲线：ρ≈40% 时 Engram 已能匹敌纯 MoE
- 最优点在 ρ≈75%-80%（分配 20-25% 给 Engram）
- 10B 规模验证损失：1.7248 (ρ=100%) → 1.7109 (ρ≈80%)，Δ=0.0139

#### 3.2 Infinite Memory Regime (p.8)
- 固定 3B MoE 骨干，扩展 Engram 槽位从 2.58×10^5 到 10^7（增加约 13B 参数）
- 对比 OverEncoding 基线
- **结果 (Figure 3 右)**: 验证损失随槽位数呈**对数线性**下降（幂律）
- Engram 比 OverEncoding 从相同记忆预算中释放更大扩展潜力

---

### 4. Large Scale Pre-training (p.8-11)

#### 4.1 Experimental Setup (p.9-10)
**数据**: 262B tokens，DeepSeek-v3 tokenizer (128k 词表)

**模型配置**:
| 模型 | 总参数 | 激活参数 | 专家数 | Engram参数 |
|------|--------|----------|--------|------------|
| Dense-4B | 4.1B | 3.8B | - | - |
| MoE-27B | 26.7B | 3.8B | 2+72 (top-6) | - |
| Engram-27B | 26.7B | 3.8B | 2+55 (top-6) | 5.7B |
| Engram-40B | 39.5B | 3.8B | 2+55 (top-6) | 18.5B |

**Engram 配置**:
- 层位置: [2, 15]
- N-gram: [2, 3]
- 头数: 8
- 维度: 1280
- 学习率: 5× 基础学习率
- 优化器: Adam（仅嵌入），无权重衰减
- 卷积参数零初始化

**骨干架构**:
- 30 层 Transformer，隐藏维度 2560
- MLA (Multi-head Latent Attention) 32 头
- mHC 扩展率 4
- 优化器: Muon

#### 4.2 Experimental Results (p.10-11, Table 1 p.9)
**Engram-27B vs MoE-27B 主要提升**:

| 类别 | 基准 | 提升 |
|------|------|------|
| 语言建模 | Pile loss | 1.960→1.950 |
| | Validation loss | 1.634→1.622 |
| 知识 | MMLU | 57.4→60.4 (+3.0) |
| | MMLU-Pro | 28.3→30.1 (+1.8) |
| | CMMLU | 57.9→61.9 (+4.0) |
| | C-Eval | 58.0→62.7 (+4.7) |
| 推理 | BBH | 50.9→55.9 (+5.0) |
| | ARC-Challenge | 70.1→73.8 (+3.7) |
| | DROP | 55.7→59.0 (+3.3) |
| 阅读理解 | RACE-High | 75.4→78.2 (+2.8) |
| 代码 | HumanEval | 37.8→40.8 (+3.0) |
| | MBPP | 46.6→48.2 (+1.6) |
| 数学 | GSM8K | 58.4→60.6 (+2.2) |
| | MATH | 28.3→30.7 (+2.4) |

**关键发现**: 推理任务提升 > 知识任务提升（违反直觉）

---

### 5. Long Context Training (p.11-12)

#### 5.1 Setup (p.11)
- 上下文扩展: YaRN (scale=10, α=1, β=32, f=0.707)
- 32768 token 上下文训练 5000 步（30B tokens）

**评估**:
- LongPPL: Book, Paper, Code, Long-CoT
- RULER: NIAH (S/MK/MV/MQ), Variable Tracking, CWE, FWE, QA

#### 5.2 Results (p.12, Table 2 p.11)
**Iso-Loss 设置** (Engram-27B 46k vs MoE-27B 50k，预训练 loss 相同):
- Multi-Query NIAH: 84.2 → 97.0
- Variable Tracking: 77.0 → 87.2
- FWE: 73.0 → 98.6

**关键发现**:
1. 长上下文能力与基础模型能力耦合，不仅取决于架构
2. Engram 用 82% 预训练 FLOPs 即可匹配基线 LongPPL，RULER 上超越

---

### 6. Analysis (p.13-18)

#### 6.1 Effective Depth Analysis (p.13-15)
**假设**: Engram 通过免除早期层的静态知识重建，等效于增加模型深度

**6.1.1 LogitLens 分析 (Figure 4a)**:
- 计算各层隐藏状态投影到输出分布与最终分布的 KL 散度
- Engram 早期层 KL 散度更低，说明预测收敛更快

**6.1.2 CKA 分析 (Figure 4b-c)**:
- 计算 Engram 与 MoE 各层表示的相似度
- 软对齐索引 a_j（公式 9）量化"有效 MoE 深度"
- **结论**: Engram 第 5 层 ≈ MoE 第 12 层，验证深度增加假设

**实体识别案例 (Table 3 p.14)**:
- "Diana, Princess of Wales" 需要 6 层逐步组合
- Engram 可通过查找直接获取

#### 6.2 Structural Ablation (p.15-16, Figure 5)
**层敏感性实验**:
- 单 Engram 模块最优位置: Layer 2
- 最佳配置: 分两个模块放置于 Layer 2 和 6

**组件消融** (从 Val Loss 1.768 基准):
| 消融 | Val Loss | 影响 |
|------|----------|------|
| w/o multi branch | ~1.775 | 显著 |
| w/o token compress | ~1.773 | 显著 |
| w/o gating | ~1.772 | 显著 |
| + 4-gram | ~1.770 | 轻微负面 |
| w/o short conv | ~1.769 | 轻微 |

#### 6.3 Sensitivity Analysis (p.16-17, Figure 6)
**实验**: 推理时完全关闭 Engram 输出

**保留性能百分比**:
- 阅读理解: 81-93%（C3 93%）— 依赖骨干注意力
- 事实知识: 29-44%（TriviaQA 29%）— 严重依赖 Engram

**结论**: Engram 是参数化知识的主要存储库

#### 6.4 System Efficiency (p.17-18, Table 4)
**实验**: 100B Engram 完全卸载到主机内存

| 基础模型 | 配置 | 吞吐量 (tok/s) |
|----------|------|----------------|
| 4B-Dense | 基线 | 9,031.62 |
| | +100B Engram | 8,858.28 (-1.9%) |
| 8B-Dense | 基线 | 6,315.52 |
| | +100B Engram | 6,140.02 (-2.8%) |

**结论**: 卸载 100B 参数开销 < 3%

#### 6.5 Gating Visualization (p.18-19, Figure 7)
**可视化**: 门控标量 α_t 的热力图
- 英文: 多 token 命名实体（"Alexander the Great", "Milky Way"）和固定短语（"By the way"）
- 中文: 成语（"四大发明"）和历史人物（"张仲景"）
- **结论**: Engram 成功识别和处理模式化语言依赖

---

### 7. Related Work (p.18-20)
- **N-gram 建模与嵌入扩展**: FastText, SuperBPE, SCONE, OverEncoding, BLT
- **MoE**: GShard, Switch Transformer, GLaM, DeepSeekMoE
- **记忆网络**: PKM, PEER, RETRO, REALM
- **知识存储机制**: FFN 作为 Key-Value 记忆, 知识神经元, ROME/MEMIT 编辑

**与先前工作的区别**:
1. 严格等参数等 FLOPs 对比（vs SCONE/OverEncoding 的非严格设置）
2. 算法-系统协同设计（深层注入支持预取，vs 仅 Layer 0）

---

### 8. Conclusion (p.20-21)
- 条件记忆是条件计算的结构性互补
- U 形稀疏分配定律指导最优配置
- 推理任务提升超过知识任务（意外发现）
- 确定性寻址实现存储-计算解耦，支持大规模部署
- **展望**: 条件记忆是下一代稀疏模型的不可或缺的建模原语

---

### Appendices (p.31-33)
- **A. 详细架构与超参数** (p.31, Table 5)
- **B. 完整基准曲线** (p.32, Figure 8): 最后 10k 步训练曲线
- **C. Tokenizer 压缩案例** (p.33, Table 6): Top-5 合并 token 示例

---

## 关键公式索引
| 公式 | 页码 | 内容 |
|------|------|------|
| (1) | p.4 | 哈希检索 |
| (2) | p.4 | 嵌入拼接 |
| (3) | p.4 | Key/Value 投影 |
| (4) | p.4 | 门控计算 |
| (5) | p.4 | 卷积输出 |
| (6) | p.5 | 多分支门控 |
| (7) | p.7 | 稀疏分配比 |
| (8) | p.14 | CKA 定义 |
| (9) | p.14 | 软对齐索引 |

---

## 关键图表索引
| 图/表 | 页码 | 内容 |
|-------|------|------|
| Figure 1 | p.3 | Engram 架构图 |
| Figure 2 | p.5 | 训练/推理系统实现 |
| Figure 3 | p.6 | 稀疏分配 U 形曲线 & 记忆扩展曲线 |
| Figure 4 | p.13 | LogitLens KL 散度 & CKA 热力图 |
| Figure 5 | p.15 | 架构消融结果 |
| Figure 6 | p.17 | Engram 关闭后保留性能 |
| Figure 7 | p.19 | 门控可视化 |
| Table 1 | p.9 | 主实验结果 |
| Table 2 | p.11 | 长上下文结果 |
| Table 3 | p.14 | 实体识别案例 |
| Table 4 | p.18 | 推理吞吐量 |
