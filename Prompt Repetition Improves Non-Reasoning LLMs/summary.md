# Prompt Repetition Improves Non-Reasoning LLMs

## 基本信息

| 项目 | 内容 |
|------|------|
| **标题** | Prompt Repetition Improves Non-Reasoning LLMs |
| **作者** | Yaniv Leviathan*, Matan Kalman* (*Equal contribution), Yossi Matias |
| **机构** | Google Research |
| **arXiv** | 2512.14982v1 |
| **发表时间** | 2025年12月 (Preprint) |
| **实验时间** | 2025年2-3月 |
| **代码链接** | 未提供 |

## 核心思想（一句话）

将用户的输入 prompt 重复一遍（`<QUERY>` -> `<QUERY><QUERY>`），就能在不增加生成 token 数和延迟的情况下，提升非推理模式下 LLM 的表现。

---

## 章节内容概要

### 1. Prompt Repetition（第1-2页）

- **问题背景**：LLM 通常以因果语言模型（causal LM）训练，past tokens 无法 attend to future tokens。因此 token 的顺序会影响预测性能，例如 `<CONTEXT> <QUESTION>` 和 `<QUESTION> <CONTEXT>` 的表现不同。
- **核心方法**：将输入从 `<QUERY>` 变为 `<QUERY><QUERY>`（简单重复一次）。这样每个 prompt token 都能 attend to 其他所有 prompt token，解决了因果注意力的单向性限制。
- **额外动机**：推理模型（如通过 RL 训练的）经常学会在回答前重复用户请求的部分内容。Prompt repetition 将这种重复移到了可并行化的 prefill 阶段，更高效。
- **关键优势**：不增加生成 token 数、不增加延迟、不改变输出格式，可以直接 drop-in 部署。

### 2. Experiments（第2页）

- **测试模型（7个）**：
  - Gemini 2.0 Flash, Gemini 2.0 Flash Lite
  - GPT-4o-mini, GPT-4o
  - Claude 3 Haiku, Claude 3.7 Sonnet
  - Deepseek V3
- **测试基准（7个）**：
  - 标准：ARC (Challenge), OpenBookQA, GSM8K, MMLU-Pro, MATH
  - 自定义：NameIndex, MiddleMatch
- **多选题配置**：question-first vs. options-first 两种顺序
- **准确率结果**：Prompt repetition 在 70 个 benchmark-model 组合中赢了 47 个，0 个输，使用 McNemar test (p < 0.1) 作为显著性标准。
- **特别亮眼**：NameIndex 任务上 Gemini 2.0 Flash-Lite 从 21.33% 提升到 97.33%。
- **推理模式下**（think step by step）：结果为中性到轻微正面（5赢1负22平）。

- **消融实验**：
  - Prompt Repetition (Verbose)：加 "Let me repeat that:" 前缀
  - Prompt Repetition x3：重复3次
  - Padding：用等长的句号填充（对照组）
  - 结果：Verbose 和 x3 表现类似或更好，Padding 无提升（证明增益来自重复而非增加输入长度）。

- **效率**：所有情况下 prompt repetition 不增加生成输出长度和延迟。唯一例外是 Anthropic 模型在非常长的请求下延迟增加（可能因为 prefill 阶段变长）。

### 3. Related Work（第3页）

- Chain of Thought (CoT) prompting：需要任务特定示例，增加生成长度和延迟。
- "Think step by step"：提升大但增加输出长度/延迟/算力。
- Shaier [2024]：只重复 question 部分，无增益。
- Springer et al. [2024]：重复输入两次可获得更好的 text embeddings。
- Xu et al. [2024]：让模型 re-read question 可提升推理能力。

### 4. Conclusion（第3页）

- Prompt repetition 在非推理模式下一致提升模型性能。
- 延迟不受影响（仅影响可并行的 prefill 阶段）。
- 不改变生成输出的长度或格式。
- 可能是非推理场景下的一个好的默认策略。

### 附录 A（第5-12页）

- **A.1** 消融和变体的详细结果图表（Figure 2, 3）
- **A.2** 推理模式下的实验结果（Figure 4）：5赢1负22平
- **A.3** 自定义任务详细描述：
  - **NameIndex**：给定 50 个名字列表，要求输出第 25 个名字
  - **MiddleMatch**：给定 40 个名字/数字列表（K=10 种，有重复），找出位于两个给定名字之间的那个
- **A.4** 各方法的查询模板示例

---

## 关键图表索引

| 图表 | 页码 | 内容描述 |
|------|------|----------|
| **Figure 1** | p.1 | 核心结果：Prompt repetition vs. baseline 准确率对比（7个模型 x 多个基准），47/70 显著胜利，0 败 |
| **Figure 2** | p.5 | 详细对比：准确率、平均/中位回复长度、平均延迟（第1部分） |
| **Figure 3** | p.6 | 详细对比：准确率、平均/中位回复长度、平均延迟（第2部分） |
| **Figure 4** | p.7 | 推理模式（think step by step）下的结果：5/28 显著胜利，1 败 |

---

## 主要实验结果

### 非推理模式（核心结果）

| 统计项 | 数值 |
|--------|------|
| 测试组合总数 | 70 (7 models x 10 benchmark configs) |
| 显著胜利（p < 0.1） | **47** |
| 显著失败 | **0** |
| 显著性检验 | McNemar test |

### 推理模式（Think step by step）

| 统计项 | 数值 |
|--------|------|
| 测试组合总数 | 28 |
| 显著胜利 | 5 |
| 显著失败 | 1 |
| 中性 | 22 |

---

## 方法变体一览

| 方法 | 模板 | 说明 |
|------|------|------|
| Baseline | `<QUERY>` | 原始输入 |
| Prompt Repetition | `<QUERY><QUERY>` | 简单重复一次 |
| Prompt Repetition (Verbose) | `<QUERY> Let me repeat that: <QUERY>` | 加过渡语重复 |
| Prompt Repetition x3 | `<QUERY> Let me repeat that: <QUERY> Let me repeat that one more time: <QUERY>` | 重复三次 |
| Padding | `<QUERY> Ignore these periods...: ...` | 用句号填充等长（对照组，无提升） |

---

## 未来方向（13条）

1. 用重复 prompt 微调模型
2. 训练推理模型使用 prompt repetition 以提高效率（模型可能学会避免自行重复）
3. 在生成过程中周期性重复最近生成的 token；探索多轮场景的适用性
4. 只在 KV-cache 中保留第二次重复（生成阶段完全无性能损失）
5. 只重复 prompt 的部分内容（尤其对长 prompt）
6. 用小模型重排 prompt 而非全部重复
7. 探索非文本模态（如图像）的适用性
8. 分析多于 2 次重复何时更有优势
9. 进一步分析重复引起的注意力模式
10. 与 selective attention 等技术结合
11. 探索与 Prefix LM 等技术的交互
12. 研究重复何时有帮助、token 表示在重复间如何变化
13. 探索有前景的变体

---

## 核心发现总结

1. **简单有效**：仅需将 prompt 重复一次，即可在非推理模式下一致提升 LLM 性能（47/70 显著胜利，0 败）。
2. **零额外开销**：不增加生成 token 数、不增加延迟（重复仅影响可并行的 prefill 阶段）、不改变输出格式。
3. **跨模型通用**：在 Gemini、GPT、Claude、Deepseek 等 7 个主流模型上均有效。
4. **原理直观**：因果 LM 中，前面的 token 无法 attend to 后面的 token；重复 prompt 使得第二份中的每个 token 都能看到完整的上下文。
5. **与推理互补**：推理模式下 prompt repetition 中性到轻微正面，因为推理模型本身就倾向于先重复问题。
6. **增益非因长度**：对照实验（Padding）证明增益来自语义重复而非单纯增加输入长度。
