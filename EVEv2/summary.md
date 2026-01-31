# EVEv2: Improved Baselines for Encoder-Free Vision-Language Models

## 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | EVEv2: Improved Baselines for Encoder-Free Vision-Language Models |
| 作者 | Haiwen Diao*, Xiaotong Li*, Yufeng Cui*, Yueze Wang* 等 |
| 机构 | DLUT, BAAI, PKU, BUPT, UCAS, CASIA |
| 发表 | ICCV 2025 (Highlight) |
| arXiv | 2502.06788 |
| 代码 | https://github.com/baaivision/EVE |
| 模型 | https://huggingface.co/BAAI/EVE-7B-HD-v2.0 |

## 核心贡献

EVEv2.0 是一个 **Encoder-free（无编码器）** 的视觉语言模型，不依赖预训练视觉编码器（如 CLIP），而是从头学习视觉感知能力。

### 主要创新点

1. **Divide-and-Conquer 架构**：完全解耦视觉和文本的处理参数（LayerNorm、FFN、Attention 的 Q/K/V/O），但 Attention 计算保持统一以实现跨模态交互
2. **DenseFusion++ 标注引擎**：基于 LLaVA-1.6 (7B) 的高质量图像描述生成器，融合多视觉专家信息
3. **四阶段渐进式训练**：从 Patch Embedding 对齐到全模型微调的系统化训练策略
4. **无损视觉编码**：两层卷积的 Patch Embedding 设计，保证像素信息完整保留

---

## 章节索引

| 章节 | 页码 | 主要内容 |
|------|------|---------|
| 1. Introduction | p.1-2 | 问题背景、encoder-free VLM 的挑战与动机 |
| 2. Related Work | p.2 | Encoder-based VLM、Encoder-free VLM、Discrete Tokenizer 综述 |
| 3. Methodology | p.3-5 | 预实验分析、模型架构、训练流程 |
| 4. Experiments | p.6-8 | 主实验结果、消融实验、数据分析 |
| 5. Limitation and Discussion | p.8 | 局限性与未来方向 |
| 6. Conclusion | p.8 | 总结 |
| Appendix | p.14-20 | 实验细节、超参数、可视化示例 |

---

## 核心概念详解

### 1. 三类视觉编码方式对比

| 方式 | 代表模型 | 特点 | 数据效率 |
|------|---------|------|---------|
| **Vision Encoder (VE)** | LLaVA, InternVL | 使用预训练 CLIP/SigLIP，已有视觉-语言对齐 | 高（少量数据即可） |
| **Discrete Tokenizer (DT)** | Chameleon, Emu3 | VQ-VAE 量化图像为离散 token | 低（量化损失信息） |
| **Encoder-free (EVE)** | Fuyu, EVE, PaliGemma | 轻量 Patch Embedding 从头学习 | 中等（需要大量数据） |

**论文发现**（Figure 2, p.3）：
- VE 初始性能最好，但依赖预训练编码器的归纳偏置
- DT 效果最差，量化导致信息丢失
- EVE 具有最强的数据扩展效率，随数据增加可逼近 VE

### 2. Divide-and-Conquer 架构

**问题**：在统一模型中同时处理视觉和文本会导致模态干扰，学习视觉会破坏 LLM 的语言知识。

**解决方案**：为每个模态设置独立的参数，但共享 Attention 计算：

```
输入: [visual_tokens | text_tokens]
      ↓
LayerNorm:  LN_v(vis) ⊕ LN_t(txt)     ← 模态解耦
      ↓
Q/K/V:      Q_v, K_v, V_v ⊕ Q_t, K_t, V_t  ← 模态解耦
      ↓
Attention:  softmax(Q @ K^T) @ V       ← 统一计算（跨模态交互）
      ↓
O 投影:     O_v ⊕ O_t                  ← 模态解耦
      ↓
LayerNorm:  LN_v ⊕ LN_t               ← 模态解耦
      ↓
FFN:        FFN_v ⊕ FFN_t             ← 模态解耦
```

**关键洞察**（Figure 2 右侧, p.3）：
- LayerNorm 是模态干扰最严重的模块
- 仅解耦 FFN（如 MoE 方式）不够，需要完全解耦

### 3. Patch Embedding 无损编码

**设计**（公式 1, p.4）：
```
x_v = Conv2(GELU(Conv1(I)))
```

| 层 | 配置 | 输入维度 | 输出维度 | 是否无损 |
|---|------|---------|---------|---------|
| Conv1 | kernel=16, stride=16, out=1024 | 16×16×3=768 | 1024 | ✓ (768<1024) |
| Conv2 | kernel=2, stride=2, out=3584 | 2×2×1024=4096 | 3584 | ✓ (原始3072<3584) |

**最终**：32×32×3=3072 像素 → 3584 维向量，信息可完整保留。

### 4. 四阶段训练流程

| 阶段 | 数据量 | 分辨率 | 可训练参数 | 目标 |
|------|--------|--------|-----------|------|
| **Stage 1** | 10M | 800² | Patch Embedding | LLM 引导的初步对齐 |
| **Stage 2.1** | 77M | 800²→1600² | PatchEmbed + Vision Layers | 视觉感知学习（LLM 冻结） |
| **Stage 2.2** | 15M | 1600² | 全部参数 | 多任务深度对齐 |
| **Stage 3** | 7.3M | 1600² | 全部参数 | 指令微调 |

**关键技术**：
- Stage 2.1 开始前，Vision Layers 从 Text Layers **复制权重**初始化
- 代码位置：`eve/train/repeat_moe.py` 第 168-171 行

### 5. DenseFusion++ 标注引擎

基于 LLaVA-1.6 (7B)，融合多视觉专家（Tagger、Detector、OCR）的输出，生成高质量详细图像描述。

**效率**：单个 8×A100 节点，每天可生成 70 万条描述。

---

## 关键图表索引

| 图表 | 页码 | 内容 |
|------|------|------|
| Figure 1 | p.2 | VLM 视觉编码方式总览 + EVE 架构演进路线图 |
| Figure 2 | p.3 | 预实验：VE/DT/EVE 数据扩展效率对比 + 权重变化分析 |
| Figure 3 | p.4 | EVEv2.0 完整架构图（Patch Embedding + DaC Layer） |
| Figure 4 | p.5 | 四阶段训练流程图 |
| Figure 5 | p.7 | 不同 EVE 变体的训练损失曲线和性能对比 |
| Figure 6 | p.7 | 不同数据源和标注引擎的效果对比 |
| Table 1 | p.5 | 各阶段训练数据详情 |
| Table 2 | p.6 | 主实验结果：与 SOTA 模型的 benchmark 对比 |

---

## 主要实验结果

### 与 SOTA 模型对比（Table 2, p.6）

| 模型 | 类型 | 参数量 | MMMU | MMBench | SEEDBench | TextVQA | ChartQA |
|------|------|--------|------|---------|-----------|---------|---------|
| LLaVA-1.5 7B | Encoder-based | 7B | 35.3 | 64.3 | 64.3 | 46.1 | 18.2 |
| LLaVA-1.6 7B | Encoder-based | 7B | 35.1 | 67.4 | 64.7 | 64.9 | 54.8 |
| Cambrian 7B | Encoder-based | 7B | 42.7 | 75.9 | 74.7 | 71.7 | 73.3 |
| Fuyu 8B | Encoder-free | 8B | 27.9 | 10.7 | 59.3 | - | - |
| EVE 7B | Encoder-free | 7B | 32.6 | 52.3 | 64.6 | 56.8 | 59.1 |
| Mono-InternVL 1.8B | Encoder-free | 1.8B | 33.7 | 65.5 | 67.4 | 72.6 | 73.7 |
| **EVEv2.0 7B** | Encoder-free | 7B | **39.3** | **66.3** | **71.4** | **71.1** | **73.9** |

**结论**：EVEv2.0 超越所有 encoder-free 模型，接近 encoder-based 模型（如 LLaVA-1.6）。

### 架构消融实验（Figure 5, p.7）

| 架构 | 描述 | 平均准确率 (8M数据) |
|------|------|-------------------|
| EVEv1.0 | Prototype，全共享参数 | ~42% |
| EVEv1.2 | Re-parameterize，低秩增量 | ~47% |
| EVEv1.5 | MoE，仅解耦 FFN | ~50% |
| **EVEv2.0** | DaC，完全解耦 | **~52%** |

---

## 核心代码结构

```
EVE/EVEv2/
├── eve/
│   ├── model/
│   │   ├── eve_arch.py                    # 多模态输入处理（token 拼接、mask 生成）
│   │   ├── multimodal_encoder/
│   │   │   └── vision_tokenizer.py        # Patch Embedding 实现
│   │   └── language_model/
│   │       └── qwen2/
│   │           └── modeling_qwen2.py      # DaC 架构核心（moe_function、DecoderLayer）
│   └── train/
│       ├── repeat_moe.py                  # Vision Layers 权重复制脚本
│       └── train.py                       # 训练主流程
└── scripts/                               # 训练启动脚本
```

### 关键代码片段

**1. Patch Embedding（vision_tokenizer.py）**
```python
self.patch_embedding = nn.Sequential(
    nn.Conv2d(3, 1024, kernel_size=16, stride=16),  # 768→1024
    nn.GELU(),
    nn.Conv2d(1024, 3584, kernel_size=2, stride=2)  # 合并4个patch→3584
)
```

**2. 模态路由函数（modeling_qwen2.py:81-84）**
```python
def moe_function(hidden_states, visual_token_mask, raw_layers, moe_layers, training):
    hidden_states = raw_layers(hidden_states) * (1. - visual_token_mask) \
                    + moe_layers(hidden_states) * visual_token_mask
    return hidden_states
```

**3. 权重复制（repeat_moe.py:168-171）**
```python
for name, param in params_moe.items():
    # moe_mlp.weight ← mlp.weight
    params_raw[name] = params_raw[name.replace('moe_', '')]
model.load_state_dict(params_raw, strict=True)
```

---

## 核心发现总结

1. **Discrete Tokenizer 效果最差**：VQ-VAE 量化导致信息丢失，即使增加数据也难以追平
2. **LayerNorm 是模态干扰的主要来源**：仅解耦 FFN 不够，需要完全解耦
3. **从 LLM 复制权重初始化至关重要**：让 Vision Layers 继承语言处理能力，加速收敛
4. **高质量标注数据显著提升效率**：DenseFusion++ 比原始 web 数据效果好得多
5. **Encoder-free 方案具有强数据扩展性**：随数据增加可持续逼近 encoder-based 模型

---

## 局限性与未来方向

1. **数据规模限制**：仅用 100M 数据，与 SOTA 模型（如 LLaVA-OV 用 10B+）有差距
2. **计算资源限制**：未能充分探索 scaling law
3. **未来方向**：
   - 模型规模扩展
   - 数据规模扩展
   - 多模态扩展（音频、视频）

---

## 快速参考

### 复现环境
```bash
git clone https://github.com/baaivision/EVE.git
cd EVE/EVEv2
pip install -e . && pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### 推理示例
```python
from eve.model.builder import load_pretrained_model
model_path = "BAAI/EVE-7B-HD-v2.0"
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base=None)
```
