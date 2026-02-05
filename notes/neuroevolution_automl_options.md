# Neuroevolution 与 AutoML 在本架构中的应用

> 本文档是 [recurrent_moe_manifold_architecture.md](./recurrent_moe_manifold_architecture.md) 的补充，记录进化算法和自动机器学习方法在架构搜索和超参数调优中的可选应用。

---

## 1. Neuroevolution 概述

**Neuroevolution**：使用进化算法（如遗传算法）来构建和优化神经网络。与梯度方法不同，它可以同时进化网络的拓扑结构和权重，且不依赖可微的目标函数。

### 1.1 NEAT 核心思想

**NEAT**（NeuroEvolution of Augmenting Topologies）是最具影响力的神经进化算法：

**原则1：从最小网络开始**
```
初始：输入直连输出（无隐藏层）
    [Input] ──→ [Output]

进化后：自动发现需要的结构
    [Input] ──→ [Hidden1] ──→ [Output]
              ↘ [Hidden2] ↗
```
避免在不必要的复杂结构上浪费搜索资源。

**原则2：结构与权重同时进化**
```
突变类型：
├── 权重突变：微调现有连接的权重
├── 添加连接：在两个未连接的节点间加边
└── 添加节点：在现有边上插入新节点
```

**原则3：历史标记（Innovation Number）**
每个新结构变异获得全局唯一的"创新号"，解决不同拓扑网络的交叉问题。

**原则4：物种形成（Speciation）**
种群分成多个"物种"，保护创新结构，给新结构成长时间。

---

## 2. Neuroevolution 在本架构中的应用

### 2.1 专家结构的进化搜索

不预定义专家结构，让进化自动发现最优结构：

```python
def evolve_expert_pool(num_experts, fitness_fn):
    # 初始：每个专家是最简单的线性层
    population = [MinimalExpert() for _ in range(num_experts)]

    for generation in range(max_generations):
        # 评估适应度（在预训练任务上的表现）
        fitness = [fitness_fn(expert) for expert in population]

        # 选择、交叉、突变
        population = select_and_mutate(population, fitness)

        # NEAT 风格的结构突变
        for expert in population:
            if random() < p_add_node:
                expert.add_hidden_node()
            if random() < p_add_connection:
                expert.add_connection()

    return population  # 结构各异、功能分化的专家池
```

**优势**：进化天然产生多样性，与"专家分化"目标契合。

### 2.2 路由策略的进化

路由决策是离散的（选哪些专家），进化算法天然处理离散空间：

```python
def evolve_routing_strategy(expert_pool, env):
    population = [RandomRoutingStrategy() for _ in range(pop_size)]

    for generation in range(max_generations):
        # 在环境中评估（累积奖励作为适应度）
        fitness = [evaluate_in_env(strategy, expert_pool, env)
                   for strategy in population]

        population = select_and_mutate(population, fitness)

    return best(population)
```

**优势**：不需要可微的奖励函数，适合 RL 环境中的策略搜索。

### 2.3 Gate 的进化优化

Gate 涉及离散决策 + 延迟奖励，进化算法避免了梯度传播的困难：

```python
# Gate 的进化搜索空间
gate_genome = {
    'network_structure': [...],      # Gate 网络的拓扑
    'threshold': 0.5,                 # 终止阈值
    'input_features': ['state_norm', 'state_delta', 'free_energy'],
}
```

### 2.4 整体架构进化（激进方案）

将 NEAT 思想应用到整个 MoE 递归系统：

```
初始配置（最小可行）：
├── 2 个专家（简单 MLP）
├── 固定路由（随机分配）
└── 固定 Gate（2 次迭代后终止）

进化后（自动发现）：
├── N 个专家（结构各异）
├── 学习到的路由模式
├── 自适应 Gate
└── 最优的流形约束配置
```

可进化的"基因组"：
- 专家数量与结构
- 路由网络结构和稀疏度
- Gate 配置（结构、阈值、输入特征）
- 流形约束权重
- 训练超参数

---

## 3. AutoML 概述

**AutoML**（Automated Machine Learning）：自动化机器学习流程中需要人类专家介入的环节，包括超参数调优、架构搜索、特征工程等。

核心组件：
- **搜索空间**：定义可能的配置
- **搜索策略**：贝叶斯优化、进化算法、强化学习等
- **性能评估**：如何高效评估候选方案

---

## 4. AutoML 在本架构中的应用

### 4.1 可自动调优的参数

| 类别 | 参数 | 推荐方法 |
|-----|------|---------|
| 流形约束 | 低维性/平滑性/正交性权重 | 贝叶斯优化 |
| 训练策略 | 阶段一迭代次数、课程学习进度 | Hyperband |
| Gate | 自由能阈值、效率惩罚 λ | TPE |
| 架构 | 专家数量、隐藏层维度 | NAS |

### 4.2 示例：用 Optuna 调优流形约束权重

```python
import optuna

def objective(trial):
    config = {
        'smoothness_weight': trial.suggest_float('smooth', 0.001, 1.0, log=True),
        'orthogonality_weight': trial.suggest_float('ortho', 0.001, 1.0, log=True),
        'dim_reduction_weight': trial.suggest_float('dim', 0.01, 10.0, log=True),
    }

    model = train_with_config(config)
    val_loss = evaluate(model)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
best_config = study.best_params
```

### 4.3 示例：用 Hyperband 高效搜索训练配置

```python
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

search_space = {
    'expert_hidden_dim': tune.choice([256, 512, 1024]),
    'num_experts': tune.choice([8, 16, 32, 64]),
    'gate_signal': tune.choice(['free_energy', 'efficiency']),
    'differentiation': tune.choice(['random', 'orthogonal', 'competitive']),
    'learning_rate': tune.loguniform(1e-5, 1e-3),
}

scheduler = HyperBandScheduler(
    max_t=100,        # 最大训练 epoch
    grace_period=10,  # 至少训练 10 epoch 再判断
    reduction_factor=3
)

analysis = tune.run(
    train_fn,
    config=search_space,
    scheduler=scheduler,
    num_samples=50
)
```

Hyperband 核心思想：早停差的配置，把资源给好的配置。

### 4.4 示例：组合搜索实验配置

```python
# 自动搜索实验矩阵中的最优组合
search_space = {
    # 维度1：专家分化机制
    'differentiation': tune.choice(['random', 'orthogonal', 'competitive']),
    'ortho_weight': tune.loguniform(0.01, 1.0),

    # 维度2：位置感知
    'position_aware': tune.choice([True, False]),

    # 维度3：Gate 训练信号
    'gate_signal': tune.choice(['free_energy_threshold', 'efficiency_constraint']),
    'gate_lambda': tune.loguniform(0.001, 0.1),
}
```

---

## 5. 推荐的结合方案

| 阶段 | 方法 | 目标 |
|-----|------|------|
| 早期验证 | AutoML（Optuna） | 快速找到可行的超参数配置 |
| 架构搜索 | Neuroevolution | 自动发现专家结构、路由模式 |
| 正式训练 | 梯度优化 | 用找到的最优配置训练权重 |

---

*创建日期：2026-02-06*
*关联文档：[主架构文档](./recurrent_moe_manifold_architecture.md)*
