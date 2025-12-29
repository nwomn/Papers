# DendAttn-UT-ACT用于世界模型推理

## 1. 架构可行性分析

### 1.1 为什么DendAttn-UT-ACT适合世界模型？

**世界模型的核心需求：**
1. **多模态感官融合**：视觉、触觉、本体感觉、听觉等
2. **持续的状态维护**：记住"世界是什么样的"
3. **自适应推理深度**：简单场景快速反应，复杂场景深度思考
4. **高效的实时推理**：智能体需要快速决策

**DendAttn-UT-ACT的优势：**
1. **状态矩阵 = 世界状态表征**
   - `S ∈ [H, d_k, d_v]` 固定大小，O(1)内存
   - 通过Gated Delta Rule持续更新世界认知
   - 类似人脑的工作记忆（Working Memory）

2. **自适应计算时间 = 思考深度**
   - 简单场景（空旷走廊）：2-3次迭代，快速前进
   - 复杂场景（障碍密集）：10-15次迭代，仔细规划
   - 类似人类的"系统1"和"系统2"思维

3. **线性注意力 = 实时处理**
   - O(L)复杂度适合实时控制（30-60 FPS）
   - 不像标准Transformer的O(L²)会随输入暴增

---

## 2. 关键问题：直接拼接 vs 编码表征

### 2.1 计算效率对比

#### 场景设置
智能体需要处理的感官输入：
- **视觉**：256×256 RGB图像 = 196,608个值
- **触觉**：10个传感器
- **本体感觉**：18个关节角度+速度 = 36个值
- **其他**：位置、朝向等 = 10个值
- **总计**：约 196,664 个原始数值

---

#### 方案A：直接拼接为固定长度向量

```python
# 直接将所有感官输入拉平
raw_input = torch.cat([
    vision.flatten(),      # 196,608
    touch.flatten(),       # 10
    proprioception.flatten(), # 36
    other.flatten()        # 10
], dim=-1)  # [B, 196,664]

# 投影到模型维度
x = linear_proj(raw_input)  # [B, 196,664] -> [B, 196,664, D=2048]

# DendAttn-UT-ACT推理
output, metrics = model(x)  # 每次迭代处理 L=196,664
```

**计算成本（每次ACT迭代）：**
- 输入长度：L = 196,664
- 线性注意力：O(L·D²) = O(196,664 × 2048²) ≈ **825 GFLOPS**
- 如果迭代10次：**8.25 TFLOPS**
- 在RTX 4090 (82 TFLOPS)上：**100ms / 次迭代**
- 10次迭代：**1秒** → 只能跑1 FPS！

**问题：**
❌ 推理太慢，无法实时控制
❌ 大量冗余信息（相邻像素高度相关）
❌ 没有利用视觉的局部性先验
❌ 内存占用巨大：196,664 × 2048 × 4 bytes = 1.6 GB（仅激活值）

---

#### 方案B：先编码为表征

```python
# 1. 多模态编码器（各自处理）
vision_encoder = PretrainedViT()  # 或 ResNet, EfficientNet
touch_encoder = MLPEncoder(10, 64)
proprio_encoder = MLPEncoder(36, 64)
other_encoder = MLPEncoder(10, 32)

# 2. 提取语义表征
vision_feat = vision_encoder(vision_img)  # [B, 1, 768] - 单个全局特征
# 或者使用patch features: [B, 16, 768] - 16个patch
touch_feat = touch_encoder(touch)         # [B, 1, 64]
proprio_feat = proprio_encoder(proprio)   # [B, 1, 64]
other_feat = other_encoder(other)         # [B, 1, 32]

# 3. 拼接表征
embeddings = torch.cat([
    vision_feat,   # [B, 1, 768]
    touch_feat,    # [B, 1, 64]
    proprio_feat,  # [B, 1, 64]
    other_feat     # [B, 1, 32]
], dim=1)  # [B, 4, 928]

# 4. 投影到统一维度
x = modality_proj(embeddings)  # [B, 4, D=2048]

# 5. DendAttn-UT-ACT推理
output, metrics = model(x)  # 每次迭代处理 L=4
```

**计算成本分解：**

**编码阶段（一次性）：**
- Vision ViT：约 1-2 GFLOPS（使用预训练模型）
- Touch/Proprio/Other MLP：约 0.01 GFLOPS
- **总计：2 GFLOPS**

**DendAttn-UT-ACT推理（每次迭代）：**
- 输入长度：L = 4（或16如果用patch features）
- 线性注意力：O(L·D²) = O(4 × 2048²) ≈ **33.5 MFLOPS**
- 如果迭代10次：**335 MFLOPS**
- 在RTX 4090上：**<1ms / 次迭代**
- 10次迭代：**约10ms**

**总推理时间：**
- 编码：2 GFLOPS ≈ 2ms
- ACT迭代：10ms
- **总计：12ms → 可以跑 83 FPS！**

---

### 2.2 效率对比总结

| 指标 | 方案A：直接拼接 | 方案B：先编码 | 加速比 |
|------|----------------|--------------|--------|
| 输入序列长度 L | 196,664 | 4-16 | **12,291×-49,166×** |
| 单次迭代计算量 | 825 GFLOPS | 0.0335 GFLOPS | **24,627×** |
| 单次迭代时间 | 100ms | <1ms | **>100×** |
| 10次ACT迭代 | 1000ms | 10ms | **100×** |
| 总推理时间 | ~1000ms | ~12ms | **83×** |
| 可达帧率 | 1 FPS | 83 FPS | **83×** |
| 内存占用 | 1.6 GB | 0.03 GB | **53×** |

**结论：先编码方案在所有维度上都碾压直接拼接！**

---

## 3. 完整架构设计：DendAttn-WorldModel

### 3.1 整体架构

```
感官输入 → 多模态编码器 → DendAttn-UT-ACT → 行为解码器 → 动作输出
   ↓            ↓                ↓                 ↓            ↓
[Raw]      [Compact]        [Reasoning]       [Policy]    [Action]
Sensors    Embeddings       + State           Logits      Commands
```

### 3.2 详细组件

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class MultiModalEncoder(nn.Module):
    """
    多模态感官编码器
    将原始感官输入压缩为紧凑的语义表征
    """
    def __init__(
        self,
        vision_encoder: str = "vit_small",  # 或 "resnet18", "efficientnet_b0"
        vision_feat_dim: int = 384,
        touch_dim: int = 10,
        proprio_dim: int = 36,
        other_dim: int = 10,
        embed_dim: int = 2048
    ):
        super().__init__()

        # 1. 视觉编码器（预训练）
        if vision_encoder == "vit_small":
            # 使用预训练的ViT-Small
            self.vision_enc = torch.hub.load('facebookresearch/dino:main',
                                             'dino_vits16', pretrained=True)
            vision_feat_dim = 384
        elif vision_encoder == "resnet18":
            import torchvision.models as models
            resnet = models.resnet18(pretrained=True)
            self.vision_enc = nn.Sequential(*list(resnet.children())[:-1])
            vision_feat_dim = 512

        # 冻结视觉编码器（可选）
        # for param in self.vision_enc.parameters():
        #     param.requires_grad = False

        # 2. 其他模态编码器
        self.touch_enc = nn.Sequential(
            nn.Linear(touch_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.proprio_enc = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.other_enc = nn.Sequential(
            nn.Linear(other_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # 3. 模态投影（统一到embed_dim）
        total_feat_dim = vision_feat_dim + 64 + 64 + 32
        self.modality_proj = nn.Linear(total_feat_dim, embed_dim)

        # 4. 位置编码（可学习）
        self.num_modalities = 4
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_modalities, embed_dim))

    def forward(
        self,
        vision: torch.Tensor,      # [B, 3, H, W]
        touch: torch.Tensor,       # [B, touch_dim]
        proprio: torch.Tensor,     # [B, proprio_dim]
        other: torch.Tensor        # [B, other_dim]
    ) -> torch.Tensor:
        """
        Returns:
            embeddings: [B, num_modalities, embed_dim]
        """
        batch_size = vision.size(0)

        # 1. 各模态编码
        vision_feat = self.vision_enc(vision)  # [B, vision_feat_dim]
        if len(vision_feat.shape) == 4:  # ResNet输出 [B, C, 1, 1]
            vision_feat = vision_feat.flatten(1)

        touch_feat = self.touch_enc(touch)      # [B, 64]
        proprio_feat = self.proprio_enc(proprio) # [B, 64]
        other_feat = self.other_enc(other)       # [B, 32]

        # 2. 拼接所有特征
        all_feats = torch.cat([
            vision_feat,    # [B, 384/512]
            touch_feat,     # [B, 64]
            proprio_feat,   # [B, 64]
            other_feat      # [B, 32]
        ], dim=-1)  # [B, total_feat_dim]

        # 3. 投影到统一维度
        embeddings = self.modality_proj(all_feats)  # [B, embed_dim]

        # 4. 重塑为序列 + 添加位置编码
        # 方式1：单个token表示所有模态（最紧凑）
        embeddings = embeddings.unsqueeze(1)  # [B, 1, embed_dim]

        # 方式2：每个模态一个token（更灵活，这里使用这种）
        embeddings = torch.stack([
            vision_feat, touch_feat, proprio_feat, other_feat
        ], dim=1)  # [B, 4, feat_dim]
        embeddings = self.modality_proj(embeddings)  # [B, 4, embed_dim]

        # 添加位置编码
        embeddings = embeddings + self.pos_embed

        return embeddings


class ActionDecoder(nn.Module):
    """
    行为解码器
    从推理输出生成具体的动作指令
    """
    def __init__(
        self,
        embed_dim: int = 2048,
        action_dim: int = 18,  # 例如18个关节的目标角度
        use_discrete: bool = False,
        num_discrete_actions: int = 10
    ):
        super().__init__()
        self.use_discrete = use_discrete

        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)

        if use_discrete:
            # 离散动作空间（如Atari游戏）
            self.action_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, num_discrete_actions)
            )
        else:
            # 连续动作空间（如机器人控制）
            self.action_mean = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, action_dim)
            )
            self.action_logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, L, embed_dim] - DendAttn输出

        Returns:
            action_dict: {
                'action_logits' or 'action_mean': [B, action_dim],
                'action_std': [B, action_dim] (仅连续动作)
            }
        """
        # 池化为单个向量
        x_pooled = x.mean(dim=1)  # [B, embed_dim]

        if self.use_discrete:
            action_logits = self.action_head(x_pooled)
            return {'action_logits': action_logits}
        else:
            action_mean = self.action_mean(x_pooled)
            action_std = torch.exp(self.action_logstd).expand_as(action_mean)
            return {
                'action_mean': action_mean,
                'action_std': action_std
            }


class DendAttnWorldModel(nn.Module):
    """
    完整的世界模型：DendAttn-UT-ACT用于具身智能

    流程：
    1. MultiModalEncoder: 感官 → 表征
    2. DendAttnUniversalACT: 表征 → 推理（维护世界状态）
    3. ActionDecoder: 推理 → 动作
    """
    def __init__(
        self,
        # 编码器配置
        vision_encoder: str = "vit_small",
        touch_dim: int = 10,
        proprio_dim: int = 36,
        other_dim: int = 10,

        # DendAttn-UT-ACT配置
        embed_dim: int = 2048,
        num_heads: int = 8,
        num_branches: int = 8,
        shared_branches: int = 1,
        topk: int = 2,
        layer_group_configs: list = [1, 2, 4],
        max_iterations: int = 20,
        ponder_loss_weight: float = 0.01,
        halt_threshold: float = 0.99,

        # 解码器配置
        action_dim: int = 18,
        use_discrete_actions: bool = False,
        num_discrete_actions: int = 10
    ):
        super().__init__()

        # 1. 多模态编码器
        self.encoder = MultiModalEncoder(
            vision_encoder=vision_encoder,
            touch_dim=touch_dim,
            proprio_dim=proprio_dim,
            other_dim=other_dim,
            embed_dim=embed_dim
        )

        # 2. DendAttn-UT-ACT推理核心
        # 这里引用之前设计的DendAttnUniversalACT类
        from dendattn_ut_act import DendAttnUniversalACT  # 假设已实现
        self.reasoning_core = DendAttnUniversalACT(
            hidden_dim=embed_dim,
            num_heads=num_heads,
            num_branches=num_branches,
            shared_branches=shared_branches,
            topk=topk,
            layer_group_configs=layer_group_configs,
            max_iterations=max_iterations,
            ponder_loss_weight=ponder_loss_weight,
            halt_threshold=halt_threshold
        )

        # 3. 行为解码器
        self.decoder = ActionDecoder(
            embed_dim=embed_dim,
            action_dim=action_dim,
            use_discrete=use_discrete_actions,
            num_discrete_actions=num_discrete_actions
        )

    def forward(
        self,
        vision: torch.Tensor,
        touch: torch.Tensor,
        proprio: torch.Tensor,
        other: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        完整前向传播

        Returns:
            action_dict: 动作输出
            info_dict: 诊断信息（ponder_loss, iterations等）
        """
        # 1. 感官编码
        embeddings = self.encoder(vision, touch, proprio, other)  # [B, 4, D]

        # 2. 自适应推理
        reasoning_out, metrics = self.reasoning_core(embeddings)  # [B, 4, D]

        # 3. 动作解码
        action_dict = self.decoder(reasoning_out)

        # 4. 返回结果
        info_dict = {
            'ponder_loss': metrics['ponder_loss'],
            'avg_iterations': metrics['avg_iterations'],
            'halt_probs': metrics['halt_probs'],
            'group_usage': metrics['group_usage']
        }

        if return_attention:
            info_dict['reasoning_features'] = reasoning_out

        return action_dict, info_dict

    def get_action(
        self,
        vision: torch.Tensor,
        touch: torch.Tensor,
        proprio: torch.Tensor,
        other: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        推理时采样动作（用于实际控制）
        """
        with torch.no_grad():
            action_dict, _ = self.forward(vision, touch, proprio, other)

            if 'action_logits' in action_dict:
                # 离散动作
                if deterministic:
                    action = action_dict['action_logits'].argmax(dim=-1)
                else:
                    action = torch.multinomial(
                        F.softmax(action_dict['action_logits'], dim=-1),
                        num_samples=1
                    ).squeeze(-1)
            else:
                # 连续动作
                if deterministic:
                    action = action_dict['action_mean']
                else:
                    dist = torch.distributions.Normal(
                        action_dict['action_mean'],
                        action_dict['action_std']
                    )
                    action = dist.sample()

            return action
```

---

## 4. 与现有世界模型的关系

### 4.1 经典世界模型对比

| 模型 | 核心机制 | 问题 | DendAttn-WorldModel的改进 |
|------|---------|------|--------------------------|
| **World Models (Ha & Schmidhuber 2018)** | VAE + MDN-RNN + Controller | RNN的O(T)顺序依赖，无法并行 | 线性注意力可并行训练，状态矩阵固定内存 |
| **DreamerV2/V3 (Hafner et al.)** | RSSM (循环状态空间模型) | 隐状态固定大小但更新复杂 | DendAttn的Gated Delta Rule更简洁高效 |
| **IRIS (Micheli et al. 2023)** | Transformer + 离散token化 | O(L²)注意力，长序列慢 | 线性注意力O(L)，ACT动态深度 |
| **MuZero** | 学习的模型用于规划 | 固定深度搜索树 | ACT自适应深度，简单场景快速响应 |

### 4.2 DendAttn-WorldModel的独特优势

1. **自适应思考深度（ACT）**
   - 简单场景：快速系统1反应
   - 复杂场景：深度系统2规划
   - 类似人类的双系统思维

2. **固定内存的世界状态（状态矩阵）**
   - 不随时间步/场景复杂度增长
   - 类似人脑的工作记忆容量限制

3. **多尺度推理（可变层组）**
   - 1层组：快速本能反应
   - 2层组：常规决策
   - 4层组：复杂策略规划

4. **生物合理性**
   - 状态矩阵 ≈ 前额叶工作记忆
   - ACT ≈ 认知控制（何时停止思考）
   - 多分支 ≈ 不同脑区的专家功能

---

## 5. 训练策略

### 5.1 模仿学习（Behavior Cloning）

```python
# 从专家演示学习
def train_imitation(model, expert_dataset, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        for batch in expert_dataset:
            vision, touch, proprio, other, expert_action = batch

            # 前向传播
            action_dict, info = model(vision, touch, proprio, other)

            # 动作损失
            if 'action_logits' in action_dict:
                action_loss = F.cross_entropy(
                    action_dict['action_logits'],
                    expert_action
                )
            else:
                action_loss = F.mse_loss(
                    action_dict['action_mean'],
                    expert_action
                )

            # 总损失（包括ponder loss）
            total_loss = action_loss + info['ponder_loss']

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### 5.2 强化学习（PPO/SAC）

```python
def train_rl(model, env, total_steps=1e6):
    # 使用PPO/SAC等RL算法
    # DendAttn-WorldModel作为策略网络

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for step in range(int(total_steps)):
        # 收集轨迹
        obs = env.reset()
        done = False
        trajectory = []

        while not done:
            action = model.get_action(
                obs['vision'], obs['touch'],
                obs['proprio'], obs['other'],
                deterministic=False
            )
            next_obs, reward, done, info = env.step(action)
            trajectory.append((obs, action, reward, done))
            obs = next_obs

        # PPO更新
        update_ppo(model, trajectory, optimizer)
```

### 5.3 世界模型学习（自监督）

```python
def train_world_model(model, interaction_data):
    """
    学习预测下一个状态
    类似DreamerV3的RSSM训练
    """
    # 添加预测头
    next_state_predictor = nn.Linear(2048, 2048)

    for batch in interaction_data:
        vision_t, touch_t, proprio_t, other_t = batch['state_t']
        vision_t1, touch_t1, proprio_t1, other_t1 = batch['state_t+1']
        action_t = batch['action_t']

        # 编码当前状态
        state_t = model.encoder(vision_t, touch_t, proprio_t, other_t)

        # 推理 + 动作条件
        reasoning_out, _ = model.reasoning_core(state_t)
        # 融合动作信息
        conditioned = reasoning_out + action_embedding(action_t)

        # 预测下一状态
        pred_state_t1 = next_state_predictor(conditioned)

        # 真实下一状态
        true_state_t1 = model.encoder(vision_t1, touch_t1, proprio_t1, other_t1)

        # 预测损失
        prediction_loss = F.mse_loss(pred_state_t1, true_state_t1)
```

---

## 6. 实验建议

### 6.1 基准测试环境

1. **简单环境（验证自适应性）**
   - CartPole（2D平衡）
   - 验证：简单任务应该使用少量迭代（2-3次）

2. **中等环境（验证效率）**
   - MuJoCo（Humanoid, Ant）
   - 对比标准Transformer的推理速度

3. **复杂环境（验证能力）**
   - Meta-World（多任务操作）
   - Isaac Gym（复杂机器人）
   - 验证：复杂任务使用更多迭代（10-15次）

### 6.2 关键指标

| 指标类别 | 具体指标 | 目标 |
|---------|---------|------|
| **性能** | 任务成功率 | 与SOTA持平或更好 |
| **效率** | 推理时间（ms/step） | <20ms（50+ FPS） |
| **自适应性** | 迭代次数分布 | 简单场景少，复杂场景多 |
| **内存** | 峰值GPU内存 | <2GB（单智能体） |
| **样本效率** | 达到80%成功率的样本数 | 与模仿学习/RL baseline对比 |

---

## 7. 潜在挑战与解决方案

### 7.1 挑战1：编码器与推理核心的耦合

**问题**：编码器预训练在ImageNet，推理核心在具身任务训练，可能不匹配

**解决方案**：
- 使用具身任务预训练的视觉编码器（如R3M, MVP）
- 或者端到端微调（用小学习率更新编码器）
- 添加适配层（adapter）连接编码器和推理核心

### 7.2 挑战2：ACT的训练稳定性

**问题**：ACT的halting network可能不稳定，导致过早停止或永不停止

**解决方案**：
- 温和的ponder loss退火（从0.01逐渐减小到0.001）
- 设置最小/最大迭代次数硬约束
- 初始化halting network的bias使初始halt概率适中（~0.1）

### 7.3 挑战3：多模态融合

**问题**：不同模态的重要性不同，简单拼接可能不够

**解决方案**：
```python
# 添加跨模态注意力
class CrossModalFusion(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8)

    def forward(self, modality_feats):
        # modality_feats: [B, num_modalities, dim]
        # 计算模态间的注意力
        fused, _ = self.cross_attn(
            modality_feats, modality_feats, modality_feats
        )
        return fused + modality_feats  # 残差连接
```

---

## 8. 总结

### 8.1 核心结论

✅ **DendAttn-UT-ACT非常适合世界模型推理**

✅ **必须使用"先编码"方案，而非直接拼接**
   - 效率提升：**83倍**
   - 可达帧率：**83 FPS** vs 1 FPS

✅ **关键优势**：
   1. 自适应思考深度（简单快，复杂慢）
   2. 固定内存的世界状态（O(1)）
   3. 实时推理能力（<20ms）
   4. 生物合理性（双系统思维）

### 8.2 建议的实现路径

**阶段1：基础实现（1-2周）**
- 实现MultiModalEncoder + ActionDecoder
- 集成DendAttn-UT-ACT
- 在CartPole验证可行性

**阶段2：效率优化（1周）**
- 对比直接拼接vs编码方案
- 测量推理时间和内存
- 优化编码器选择

**阶段3：性能验证（2-3周）**
- MuJoCo环境训练
- 对比标准Transformer/LSTM baselines
- 分析ACT的自适应性

**阶段4：复杂环境（2-4周）**
- Meta-World多任务
- 验证长期记忆能力
- 发表结果

---

## 9. 与大脑的类比（再次强化）

| 大脑机制 | DendAttn-WorldModel | 功能 |
|---------|---------------------|------|
| **感官皮层** | MultiModalEncoder | 将原始感官信号转为神经表征 |
| **工作记忆** | 状态矩阵 S | 维持当前任务相关的"世界模型" |
| **前额叶** | DendAttn-UT-ACT | 推理、规划、决策 |
| **运动皮层** | ActionDecoder | 生成具体的肌肉控制信号 |
| **认知控制** | ACT | 决定何时停止思考、开始行动 |
| **多脑区协作** | 多分支路由 | 不同任务激活不同"脑区组合" |

这个架构不仅高效，而且具有深刻的生物学启发！
