# Rational ANOVA Networks (RAN)

## 基本信息

| 项目 | 内容 |
|------|------|
| **标题** | Rational ANOVA Networks |
| **作者** | Jusheng Zhang, Ningyuan Liu, Qinhan Lyu, Jing Yang, Keze Wang |
| **机构** | Sun Yat-sen University, China |
| **arXiv** | 2602.04006v1 |
| **日期** | February 5, 2026 (Preprint) |
| **代码** | https://github.com/jushengzhang/Rational-ANOVA-Networks.git |

---

## 核心思想

RAN 是一种结合 **函数型 ANOVA 分解** 和 **Padé 风格有理逼近** 的神经网络架构，使非线性本身变得可学习，同时保持深度训练的稳定性和可控性。

### 核心公式 (Eq. 1-2, p.1-2)

$$f(\mathbf{x}) \approx \sum_{i=1}^{d} r_i(x_i) + \sum_{(i,j)\in S} r_{ij}(x_i, x_j)$$

- **主效应 (Main Effects)**: $\sum r_i(x_i)$ — 单变量有理函数
- **成对交互 (Pairwise Effects)**: $\sum r_{ij}(x_i, x_j)$ — 稀疏的二元有理函数

---

## 章节概要与页号索引

| 章节 | 页码 | 内容概要 |
|------|------|----------|
| **Abstract** | p.1 | 提出 RAN 架构，结合 ANOVA 分解和有理逼近，在视觉任务上超越 MLP 和 KAN |
| **1. Introduction** | p.1-2 | 动机：固定激活函数限制可解释性；KAN 存在计算效率和边界稳定性问题 |
| **2. Rational-ANOVA Networks** | p.2-5 | 架构详解：ANOVA 结构、可学习有理单元、残差门控、深度兼容性 |
| **2.1 ANOVA-Induced Architecture** | p.2 | 低阶函数分解，可控交互集 $S$ |
| **2.2 Learnable Rational Units** | p.3 | 1D/2D 有理单元，分母强制正定避免极点 |
| **2.3 Deep Compatibility via Residual Gating** | p.4 | 残差门控 $r(x) = x + \alpha \cdot (\tilde{r}(x) - x)$，确保身份初始化 |
| **2.4 RAN as a Drop-in FFN Replacement** | p.4 | 可直接替换 Transformer 中的 FFN |
| **2.5 Parameter Budgeting** | p.4-5 | 参数量计算：$\text{Params} \approx d(m+n+2) + |S|(T+S+1)$ |
| **3. Learning Dynamics** | p.5-6 | eNTK 分析、结构化核分解、有理单元深度稳定性证明 |
| **4. Experiments** | p.6-8 | 视觉分类基准、ViT 集成、PolyU 去噪、消融实验 |
| **5. Related Work** | p.8 | 可学习非线性、KAN、加性模型相关工作 |
| **6. Conclusion** | p.8-9 | 总结贡献、影响声明、局限性 |
| **References** | p.9-11 | 参考文献 |
| **Supplementary** | p.12-27 | 扩展基准、理论证明、消融实验 |

---

## 关键公式索引

| 公式 | 页码 | 内容 |
|------|------|------|
| Eq. 1-2 | p.1-2 | **ANOVA 分解结构**: $f(\mathbf{x}) = \sum r_i(x_i) + \sum r_{ij}(x_i, x_j)$ |
| Eq. 4 | p.3 | **1D 有理单元**: $\tilde{r}_i(x) = p_i(x)/d_i(x)$, $d_i(x) = 1 + \text{softplus}(\cdot) + \varepsilon$ |
| Eq. 5 | p.3 | **2D 有理单元**: $\tilde{r}_{ij}(x,y) = p_{ij}(x,y)/d_{ij}(x,y)$ |
| Eq. 6 | p.4 | **残差门控**: $r(x) = x + \alpha \cdot (\tilde{r}(x) - x)$ |
| Eq. 7 | p.4 | **门控雅可比**: $\partial r/\partial x = (1-\alpha) + \alpha \cdot \partial\tilde{r}/\partial x$ |
| Eq. 8 | p.4 | **RAN-FFN**: $\text{RAN-FFN}(h) = W_2 r(W_1 h) + b$ |
| Eq. 10 | p.4 | **参数量公式**: $\text{Params} \approx d(m+n+2) + |S|(T+S+1)$ |
| Eq. 14 | p.5 | **eNTK 核**: $K_t(x_o, x_u) = \langle \nabla_\theta z_\theta(x_o), \nabla_\theta z_\theta(x_u) \rangle^\top$ |
| Eq. 17 | p.5 | **结构化核分解**: $K_t^{\text{rat}} = \sum K_{t,i}^{\text{main}} + \sum K_{t,ij}^{\text{pair}}$ |
| Eq. 18 | p.5 | **有理单元 Lipschitz 界**: $|\partial\tilde{r}/\partial x| \leq |p'(x)| + |p(x)||d'(x)|$ |

---

## 关键图表索引

| 图表 | 页码 | 内容描述 |
|------|------|----------|
| **Figure 1** | p.2 | **MLP vs KAN vs RAN 对比图**: MLP 固定激活、KAN 边上样条、RAN 可学习有理单元 + ANOVA 拓扑 |
| **Figure 2** | p.3 | **深度 RAN 架构图**: 残差块堆叠、1D/2D 有理单元结构、scatter-add 操作 |
| **Figure 3** | p.4 | **学习动态对比**: (a) MLP 密集纠缠 (b) RAN ANOVA 局部性 (c) 有理稳定性 vs 多项式振荡 |
| **Figure 4** | p.7 | **PolyU 去噪效率图**: PSNR vs 参数量，RAN 位于 Pareto 前沿 |
| **Table 1** | p.7 | **多数据集精度对比**: MNIST/FMNIST/CIFAR/SVHN/EMNIST，RAN 全面领先 |
| **Table 2** | p.7 | **ViT-T/16 (ImageNet-1K)**: RAN 74.2% vs MLP 72.3%，+1.9% 提升 |
| **Table 3** | p.8 | **消融实验**: Dense vs Sparse / ReLU vs Rational 的组合对比 |

---

## 主要实验结果

### Table 1: 视觉分类基准 (参数匹配)

| Dataset | Params | KAF | MLP | KAN | **RAN** |
|---------|--------|-----|-----|-----|---------|
| MNIST | 50k | 97.45 | **97.60** | 96.50 | **97.55** |
| MNIST | 400k | 98.65 | 98.70 | 97.65 | **98.75** |
| FMNIST | 50k | 88.00 | 88.50 | 86.00 | **95.39** |
| CIFAR-10 | 1.0M | 56.95 | 56.45 | 43.32 | **59.05** |
| CIFAR-100 | 1.0M | 26.75 | 27.10 | 14.80 | **28.12** |

### Table 2: ViT-T/16 on ImageNet-1K

| Method | Mechanism | Params | FLOPs | Top-1 (%) |
|--------|-----------|--------|-------|-----------|
| MLP (Baseline) | Linear + GELU | 5.7M | 1.08G | 72.3 |
| KAN | B-Spline | OOM | OOM | - |
| KAF | Kernel Function | 5.9M | 1.12G | 73.2 |
| **RAN (Ours)** | **Rational** | **5.7M** | **1.08G** | **74.2** |

### Table 3: 消融实验 (CIFAR-10, ~1M params)

| Model | Topology | Activation | TabArena | C-10 (%) |
|-------|----------|------------|----------|----------|
| MLP (Base) | Dense | ReLU | 0.82 | 56.0 |
| MLP-Rat | Dense | Rational | 0.89 | 56.8 |
| ANOVA-ReLU | Sparse | ReLU | 0.86 | 55.2 |
| **RAN (Ours)** | **Sparse** | **Rational** | **0.96** | **58.3** |

---

## 核心贡献

1. **架构创新**: 首次将函数型 ANOVA 分解与 Padé 有理逼近结合，提出显式低阶交互拓扑
2. **稳定性保证**:
   - 分母 $d(x) \geq 1 + \varepsilon$ 避免极点
   - 残差门控确保近恒等初始化
   - 理论证明深度训练稳定性 (Appendix K, L)
3. **即插即用**: 可直接替换 Transformer/ViT 中的 FFN 层
4. **效率优势**: 参数匹配条件下超越 MLP 和 KAN，KAN 在高维设置下 OOM

---

## 与 KAN 的关键区别

| 特性 | KAN | RAN |
|------|-----|-----|
| 非线性位置 | 边上 (edge) | 节点上 (node) |
| 基函数 | B-样条 | 有理函数 (Padé) |
| 边界稳定性 | 振荡 (Runge 现象) | 稳定 (分母控制) |
| 高维可扩展性 | OOM | 可扩展 |
| 交互控制 | 隐式 | 显式 ANOVA 拓扑 |

---

## 局限性 (作者声明)

- 计算资源限制，未在大规模模型 (LLM, 多模态) 上验证
- 未在前沿训练规模下测试
- 低阶分解的归纳偏置可能与数据集偏差交互

---

## 补充材料概览 (p.12-27)

| Part | Section | 内容 |
|------|---------|------|
| I | A, M | TabArena 基准 (SOTA, win rate > 0.95)、Feynman 符号回归 |
| II | B, C, E, G | 物理启发案例: Lorentzian 势、外推稳定性、Van der Waals 发现 |
| III | D, J | 可视化训练动态、拓扑消融 |
| IV | K, L | 理论证明: 全局全纯性、Lipschitz 界、深度稳定性 |
| V | F, I | 工作流对比、超参数配置 |

---

## 官方代码实现

### 代码结构

```
code/
├── run_MINIST_FMNIST_CIFAR-10_EMNIST-Let_SVH_CIFAR-100.py  # 核心 RAN 实现 + 视觉基准
├── vit_ran.py           # ViT FFN 替换实现
├── TAB_model.py         # 表格数据模型
├── TAB_dataload.py      # 表格数据加载
├── TAB_run.py           # TabArena 运行脚本
├── runPolyU.py          # PolyU 去噪实验
├── pinn.py              # 物理启发神经网络 (Lorentzian)
├── configs/
│   └── config_for_7.yaml  # ViT 训练配置模板
├── reproduce.md         # 复现说明
└── requirements.txt     # 依赖列表
```

### 论文公式 ↔ 代码映射

| 论文内容 | 代码位置 |
|----------|----------|
| Eq. 4-5: 1D/2D 有理单元 | `run_MINIST_*.py`: `RationalGroup1D`, `RationalGroup2D` |
| Eq. 6-7: 残差门控 | `alpha` 参数，`x + alpha*(r-x)` |
| Eq. 2-3: ANOVA 拓扑 | `DeepRationalANOVA` 类 |
| Eq. 10: 参数预算 | `estimate_K()` 函数 |
| Eq. 8: FFN 替换 | `vit_ran.py`: `replace_vit_mlp_with_rational()` |

---

### 核心类实现

#### 1. RationalGroup1D (1D 有理单元)

```python
# 文件: run_MINIST_*.py:25-58
class RationalGroup1D(nn.Module):
    """
    r(x) = P(x) / (1 + softplus(Q(x)))   # 分母恒正
    output = x + α * (r(x) - x)          # 残差门控

    P(x) = p0 + p1*x + p2*x² + p3*x³     # 分子: 3阶多项式
    Q(x) = q1*x + q2*x²                  # 分母: 2阶多项式
    """
    def __init__(self, num_features, eps=1e-3, init_alpha=0.0):
        # p: (N, 4) 分子系数
        # q: (N, 2) 分母系数
        # alpha: (N,) 残差门控
        # 初始化: p1=1, 其他=0 → 近恒等映射

    def forward(self, x):
        p_out = p0 + p1*x + p2*x² + p3*x³
        den = 1.0 + F.softplus(q1*x + q2*x²)  # ≥ 1，避免极点
        r = p_out / den
        return x + self.alpha * (r - x)
```

#### 2. RationalGroup2D (2D 有理单元)

```python
# 文件: run_MINIST_*.py:64-100
class RationalGroup2D(nn.Module):
    """
    r(x,y) = P(x,y) / (1 + softplus(Q(x,y)))
    output = base + β * (r - base),  base = (x+y)/2

    P(x,y) = p0 + p1*x + p2*y + p3*x² + p4*xy + p5*y²  # 6项
    Q(x,y) = q1*x + q2*y + q3*x² + q4*xy + q5*y²       # 5项
    """
```

#### 3. DeepRationalANOVA (深度 ANOVA 架构)

```python
# 文件: run_MINIST_*.py:110-181
class DeepRationalANOVA(nn.Module):
    """
    两层残差块结构:

    Block 1 (Update):
        - main1: N 个 1D 有理单元 (主效应)
        - pair1: K 个 2D 有理单元 (稀疏成对交互)
        - scatter-add: 将成对信息写回特征向量

    Block 2 (Head):
        - main2 + pair2 → concat → Linear → 输出

    稀疏交互集 S: 随机采样 K 对 (i,j)，种子固定保证可复现
    """
```

#### 4. ViT FFN 替换

```python
# 文件: vit_ran.py:96-129
class RationalMlp(nn.Module):
    """替换 ViT 的 FFN，保留 fc1/fc2 权重"""
    def forward(self, x):
        x = self.fc1(x)        # 保留预训练权重
        x = self.act(x)        # RationalGroup1D (替换 GELU)
        x = self.fc2(x)        # 保留预训练权重
        return x

def replace_vit_mlp_with_rational(model):
    """遍历所有 block，替换 MLP 层"""
    for blk in model.blocks:
        new_mlp.fc1.weight.copy_(old_mlp.fc1.weight)  # 复制权重
        blk.mlp = new_mlp
```

---

### 关键实现细节

| 特性 | 实现方式 | 代码位置 |
|------|----------|----------|
| **正定分母** | `1 + softplus(Q) + ε` | `run_MINIST_*.py:50-51` |
| **残差门控** | `x + α*(r-x)`, α 初始为 0 | `run_MINIST_*.py:54` |
| **近恒等初始化** | `p1=1`, 其他系数=0 | `run_MINIST_*.py:37-38` |
| **稀疏交互集** | 固定种子随机采样 K 对 | `run_MINIST_*.py:118-127` |
| **参数预算匹配** | `K = (target - base) / perK` | `run_MINIST_*.py:292-296` |
| **分母慢学习率** | `lr_q = lr * 0.7` | `run_MINIST_*.py:335` |
| **ViT GELU 拟合初始化** | 预计算系数 | `vit_ran.py:46-48` |

---

### 运行示例

```bash
# 安装依赖
pip install -r requirements.txt

# 1. 视觉分类基准 (Table 1)
python run_MINIST_FMNIST_CIFAR-10_EMNIST-Let_SVH_CIFAR-100.py \
    --dataset CIFAR10 --target-params 1000000 --seed 42

# 2. ViT on ImageNet (Table 2)
# 先编辑 configs/config_for_7.yaml 设置 ImageNet 路径
python vit_ran.py

# 3. PolyU 去噪 (Figure 4)
python runPolyU.py

# 4. TabArena (Figure 5)
python TAB_run.py

# 5. Lorentzian 势 (Appendix B)
python pinn.py
```

---

### 依赖环境

```
torch>=2.1
torchvision>=0.16
numpy
pyyaml
tqdm
timm
accelerate
opencv-python
scikit-learn
matplotlib
pandas
autogluon.tabular
```
