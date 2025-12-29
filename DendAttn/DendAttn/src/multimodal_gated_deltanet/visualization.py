# -*- coding: utf-8 -*-
# MultimodalGatedDeltaNet Visualization Hooks
# 用于观察网络训练和推理时的内部状态

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


class StateCapture:
    """捕获并存储模型内部状态的钩子管理器"""

    def __init__(self, max_history: int = 100):
        """
        Args:
            max_history: 最大保存的历史步数
        """
        self.max_history = max_history
        self.reset()
        self._hooks = []

    def reset(self):
        """重置所有捕获的数据"""
        # 状态矩阵 S 的历史
        self.state_history: List[torch.Tensor] = []

        # 门控值历史
        self.beta_history: List[torch.Tensor] = []  # 输入门
        self.g_history: List[torch.Tensor] = []      # 衰减门

        # 输出权重历史
        self.output_weights_history: List[torch.Tensor] = []

        # 更新掩码历史
        self.update_mask_history: List[torch.Tensor] = []

        # 专家输出历史
        self.expert_outputs_history: List[torch.Tensor] = []

        # 梯度历史
        self.gradients: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # 激活值历史
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # 训练指标
        self.loss_history: List[float] = []
        self.grad_norm_history: List[float] = []

        # 步数计数
        self.step_count = 0

    def _trim_history(self, history_list: List):
        """保持历史记录在最大限制内"""
        while len(history_list) > self.max_history:
            history_list.pop(0)

    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class MultimodalVisualizationHooks(StateCapture):
    """针对 MultimodalGatedDeltaNet 的专用可视化钩子"""

    def __init__(self, max_history: int = 100, capture_every_n_steps: int = 1):
        """
        Args:
            max_history: 最大保存的历史步数
            capture_every_n_steps: 每隔多少步捕获一次（用于减少内存）
        """
        super().__init__(max_history)
        self.capture_every_n_steps = capture_every_n_steps
        self.layer_hooks = {}

    def register_hooks(self, model: nn.Module, layer_indices: Optional[List[int]] = None):
        """
        为模型注册可视化钩子

        Args:
            model: MultimodalGatedDeltaNet 模型
            layer_indices: 要监控的层索引，None 表示所有层
        """
        self.remove_hooks()

        # 找到所有 MultimodalGatedDeltaNet 层
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'MultimodalGatedDeltaNet':
                # 提取层索引
                layer_idx = getattr(module, 'layer_idx', 0)

                if layer_indices is not None and layer_idx not in layer_indices:
                    continue

                # 注册前向钩子
                hook = module.register_forward_hook(
                    self._create_forward_hook(name, layer_idx)
                )
                self._hooks.append(hook)

                # 注册梯度钩子
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        hook = param.register_hook(
                            self._create_grad_hook(f"{name}.{param_name}")
                        )
                        self._hooks.append(hook)

                self.layer_hooks[layer_idx] = name
                print(f"Registered hooks for layer {layer_idx}: {name}")

    def _create_forward_hook(self, name: str, layer_idx: int):
        """创建前向传播钩子"""
        def hook(module, input, output):
            if self.step_count % self.capture_every_n_steps != 0:
                return

            with torch.no_grad():
                # output 是 (o, attn_weights, past_key_values, router_logits)
                o, _, past_kv, _ = output

                # 捕获输出权重
                if hasattr(module, 'output_weights'):
                    weights = torch.softmax(module.output_weights, dim=0)
                    self.output_weights_history.append(weights.detach().cpu().clone())
                    self._trim_history(self.output_weights_history)

                # 捕获状态矩阵（如果有）
                if past_kv is not None and len(past_kv) > layer_idx:
                    state = past_kv[layer_idx]
                    if isinstance(state, dict) and 'recurrent_state' in state:
                        recurrent_state = state['recurrent_state']
                        self.state_history.append(recurrent_state.detach().cpu().clone())
                        self._trim_history(self.state_history)

                # 捕获激活值统计
                self.activations[f"{name}_output"].append({
                    'mean': o.mean().item(),
                    'std': o.std().item(),
                    'max': o.max().item(),
                    'min': o.min().item(),
                })
                self._trim_history(self.activations[f"{name}_output"])

        return hook

    def _create_grad_hook(self, name: str):
        """创建梯度钩子"""
        def hook(grad):
            if grad is not None and self.step_count % self.capture_every_n_steps == 0:
                self.gradients[name].append({
                    'norm': grad.norm().item(),
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                })
                self._trim_history(self.gradients[name])
            return grad
        return hook

    def capture_gates(self, beta: torch.Tensor, g: torch.Tensor, update_mask: torch.Tensor):
        """
        手动捕获门控值（需要在 forward 中调用）

        Args:
            beta: 输入门 [E, B, L, H]
            g: 衰减门 [E, B, L, H]
            update_mask: 更新掩码 [E, B, L, H]
        """
        if self.step_count % self.capture_every_n_steps != 0:
            return

        with torch.no_grad():
            self.beta_history.append(beta.detach().cpu().clone())
            self.g_history.append(g.detach().cpu().clone())
            self.update_mask_history.append(update_mask.detach().cpu().clone())

            self._trim_history(self.beta_history)
            self._trim_history(self.g_history)
            self._trim_history(self.update_mask_history)

    def capture_expert_outputs(self, expert_outputs: torch.Tensor):
        """
        捕获专家输出（聚合前）

        Args:
            expert_outputs: [E, B, L, H, d]
        """
        if self.step_count % self.capture_every_n_steps != 0:
            return

        with torch.no_grad():
            # 只保存统计信息，避免内存爆炸
            # 将 [E, B, L, H, d] reshape 为 [E, -1] 然后计算统计
            E = expert_outputs.shape[0]
            flat = expert_outputs.reshape(E, -1)
            stats = {
                'per_expert_mean': flat.mean(dim=1).cpu().tolist(),
                'per_expert_std': flat.std(dim=1).cpu().tolist(),
                'per_expert_norm': flat.norm(dim=1).cpu().tolist(),
            }
            self.expert_outputs_history.append(stats)
            self._trim_history(self.expert_outputs_history)

    def record_loss(self, loss: float):
        """记录损失值"""
        self.loss_history.append(loss)
        self._trim_history(self.loss_history)

    def record_grad_norm(self, model: nn.Module):
        """记录总梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norm_history.append(total_norm)
        self._trim_history(self.grad_norm_history)

    def step(self):
        """增加步数计数"""
        self.step_count += 1


class MultimodalVisualizer:
    """可视化绘图工具"""

    def __init__(self, hooks: MultimodalVisualizationHooks, save_dir: str = './vis_output'):
        """
        Args:
            hooks: 可视化钩子实例
            save_dir: 保存目录
        """
        self.hooks = hooks
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        self.expert_names = ['Shared', 'Text', 'Vision']
        self.expert_colors = ['#2ecc71', '#3498db', '#e74c3c']

    def plot_output_weights(self, save: bool = True, show: bool = False) -> Optional[plt.Figure]:
        """
        绘制输出权重随训练的变化

        Returns:
            matplotlib Figure 对象
        """
        if not self.hooks.output_weights_history:
            print("No output weights history available")
            return None

        # 获取所有历史权重 [steps, E, H]
        weights = torch.stack(self.hooks.output_weights_history).numpy()
        steps = np.arange(len(weights))
        num_experts, num_heads = weights.shape[1], weights.shape[2]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：每个专家的平均权重变化
        ax1 = axes[0]
        for e in range(num_experts):
            mean_weight = weights[:, e, :].mean(axis=1)
            ax1.plot(steps, mean_weight, label=self.expert_names[e],
                    color=self.expert_colors[e], linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Average Weight')
        ax1.set_title('Expert Output Weights Over Training')
        ax1.legend()
        ax1.set_ylim(0, 1)

        # 右图：最新权重的热力图
        ax2 = axes[1]
        latest_weights = weights[-1]  # [E, H]
        sns.heatmap(latest_weights, ax=ax2, annot=True, fmt='.3f',
                   xticklabels=[f'Head {i}' for i in range(num_heads)],
                   yticklabels=self.expert_names,
                   cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_title(f'Current Output Weights (Step {len(weights)})')

        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / 'output_weights.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_gate_distribution(self, step_idx: int = -1, save: bool = True,
                               show: bool = False) -> Optional[plt.Figure]:
        """
        绘制门控值分布

        Args:
            step_idx: 要可视化的步数索引，-1 表示最新
        """
        if not self.hooks.beta_history or not self.hooks.g_history:
            print("No gate history available. Call capture_gates() in forward pass.")
            return None

        beta = self.hooks.beta_history[step_idx].numpy()  # [E, B, L, H]
        g = self.hooks.g_history[step_idx].numpy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Beta 分布（每个专家）
        for e in range(min(3, beta.shape[0])):
            ax = axes[0, e]
            data = beta[e].flatten()
            ax.hist(data, bins=50, color=self.expert_colors[e], alpha=0.7, edgecolor='black')
            ax.set_title(f'β Distribution - {self.expert_names[e]}')
            ax.set_xlabel('β value')
            ax.set_ylabel('Frequency')
            ax.axvline(data.mean(), color='red', linestyle='--', label=f'mean={data.mean():.3f}')
            ax.legend()

        # g 分布（每个专家）
        for e in range(min(3, g.shape[0])):
            ax = axes[1, e]
            data = g[e].flatten()
            ax.hist(data, bins=50, color=self.expert_colors[e], alpha=0.7, edgecolor='black')
            ax.set_title(f'g (Decay) Distribution - {self.expert_names[e]}')
            ax.set_xlabel('g value')
            ax.set_ylabel('Frequency')
            ax.axvline(data.mean(), color='red', linestyle='--', label=f'mean={data.mean():.3f}')
            ax.legend()

        plt.suptitle(f'Gate Distributions at Step {self.hooks.step_count}', fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / f'gate_distribution_step{self.hooks.step_count}.png',
                       dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_gate_heatmap(self, step_idx: int = -1, batch_idx: int = 0, head_idx: int = 0,
                         save: bool = True, show: bool = False) -> Optional[plt.Figure]:
        """
        绘制门控值的序列热力图

        Args:
            step_idx: 步数索引
            batch_idx: batch 索引
            head_idx: head 索引
        """
        if not self.hooks.beta_history:
            print("No gate history available")
            return None

        beta = self.hooks.beta_history[step_idx].numpy()  # [E, B, L, H]
        g = self.hooks.g_history[step_idx].numpy()

        fig, axes = plt.subplots(2, 1, figsize=(14, 6))

        # Beta 热力图 [E, L]
        beta_seq = beta[:, batch_idx, :, head_idx]  # [E, L]
        ax1 = axes[0]
        im1 = ax1.imshow(beta_seq, aspect='auto', cmap='YlOrRd')
        ax1.set_yticks(range(len(self.expert_names)))
        ax1.set_yticklabels(self.expert_names)
        ax1.set_xlabel('Sequence Position')
        ax1.set_title(f'β (Input Gate) - Batch {batch_idx}, Head {head_idx}')
        plt.colorbar(im1, ax=ax1)

        # g 热力图 [E, L]
        g_seq = g[:, batch_idx, :, head_idx]  # [E, L]
        ax2 = axes[1]
        im2 = ax2.imshow(g_seq, aspect='auto', cmap='RdYlBu_r')
        ax2.set_yticks(range(len(self.expert_names)))
        ax2.set_yticklabels(self.expert_names)
        ax2.set_xlabel('Sequence Position')
        ax2.set_title(f'g (Decay Gate) - Batch {batch_idx}, Head {head_idx}')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / f'gate_heatmap_step{self.hooks.step_count}.png',
                       dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_training_curves(self, save: bool = True, show: bool = False) -> Optional[plt.Figure]:
        """绘制训练曲线（损失和梯度范数）"""
        if not self.hooks.loss_history:
            print("No training history available")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        ax1 = axes[0]
        ax1.plot(self.hooks.loss_history, color='blue', linewidth=1.5)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        if len(self.hooks.loss_history) > 10:
            # 添加移动平均
            window = min(50, len(self.hooks.loss_history) // 5)
            if window > 1:
                ma = np.convolve(self.hooks.loss_history, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.hooks.loss_history)), ma,
                        color='red', linewidth=2, label=f'MA({window})')
                ax1.legend()

        # 梯度范数曲线
        ax2 = axes[1]
        if self.hooks.grad_norm_history:
            ax2.plot(self.hooks.grad_norm_history, color='green', linewidth=1.5)
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Gradient Norm')
            ax2.set_title('Gradient Norm')
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No gradient norm data', ha='center', va='center',
                    transform=ax2.transAxes)

        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_gradient_flow(self, save: bool = True, show: bool = False) -> Optional[plt.Figure]:
        """绘制各层梯度流"""
        if not self.hooks.gradients:
            print("No gradient data available")
            return None

        fig, ax = plt.subplots(figsize=(14, 6))

        # 获取所有参数的最新梯度范数
        param_names = []
        grad_norms = []

        for name, grad_list in self.hooks.gradients.items():
            if grad_list:
                param_names.append(name.split('.')[-1])  # 简化名称
                grad_norms.append(grad_list[-1]['norm'])

        if not param_names:
            print("No gradient data to plot")
            return None

        # 按梯度范数排序
        sorted_indices = np.argsort(grad_norms)[::-1]
        param_names = [param_names[i] for i in sorted_indices[:20]]  # 只显示前20个
        grad_norms = [grad_norms[i] for i in sorted_indices[:20]]

        bars = ax.barh(range(len(param_names)), grad_norms, color='steelblue')
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names)
        ax.set_xlabel('Gradient Norm')
        ax.set_title('Gradient Flow (Top 20 Parameters)')
        ax.set_xscale('log')

        # 标记可能的问题
        for i, (bar, norm) in enumerate(zip(bars, grad_norms)):
            if norm > 10:
                bar.set_color('red')
            elif norm < 1e-6:
                bar.set_color('orange')

        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / 'gradient_flow.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_expert_contributions(self, save: bool = True, show: bool = False) -> Optional[plt.Figure]:
        """绘制专家贡献统计"""
        if not self.hooks.expert_outputs_history:
            print("No expert output history available")
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 提取统计数据
        steps = range(len(self.hooks.expert_outputs_history))
        means = np.array([h['per_expert_mean'] for h in self.hooks.expert_outputs_history])
        stds = np.array([h['per_expert_std'] for h in self.hooks.expert_outputs_history])
        norms = np.array([h['per_expert_norm'] for h in self.hooks.expert_outputs_history])

        # Mean
        ax1 = axes[0]
        for e in range(min(3, means.shape[1])):
            ax1.plot(steps, means[:, e], label=self.expert_names[e],
                    color=self.expert_colors[e], linewidth=2)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Mean Output')
        ax1.set_title('Expert Output Mean')
        ax1.legend()

        # Std
        ax2 = axes[1]
        for e in range(min(3, stds.shape[1])):
            ax2.plot(steps, stds[:, e], label=self.expert_names[e],
                    color=self.expert_colors[e], linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Std')
        ax2.set_title('Expert Output Std')
        ax2.legend()

        # Norm
        ax3 = axes[2]
        for e in range(min(3, norms.shape[1])):
            ax3.plot(steps, norms[:, e], label=self.expert_names[e],
                    color=self.expert_colors[e], linewidth=2)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Norm')
        ax3.set_title('Expert Output Norm')
        ax3.legend()

        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / 'expert_contributions.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_state_evolution(self, head_idx: int = 0, save: bool = True,
                            show: bool = False) -> Optional[plt.Figure]:
        """
        绘制状态矩阵 S 的演化

        Args:
            head_idx: 要可视化的 head 索引
        """
        if not self.hooks.state_history:
            print("No state history available")
            return None

        # state shape: [B, H, d_k, d_v]
        # 我们可视化某个 head 的状态矩阵随时间的变化

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # 取最近的几个状态
        num_states = min(6, len(self.hooks.state_history))
        indices = np.linspace(0, len(self.hooks.state_history)-1, num_states, dtype=int)

        for idx, (ax, state_idx) in enumerate(zip(axes.flat, indices)):
            state = self.hooks.state_history[state_idx]  # [B, H, d_k, d_v]
            # 取第一个 batch, 指定的 head
            state_matrix = state[0, head_idx].numpy()  # [d_k, d_v]

            im = ax.imshow(state_matrix, aspect='auto', cmap='RdBu_r',
                          norm=Normalize(vmin=-np.abs(state_matrix).max(),
                                        vmax=np.abs(state_matrix).max()))
            ax.set_title(f'Step {state_idx}')
            ax.set_xlabel('d_v')
            ax.set_ylabel('d_k')
            plt.colorbar(im, ax=ax)

        plt.suptitle(f'State Matrix Evolution (Head {head_idx})', fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / f'state_evolution_head{head_idx}.png',
                       dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def create_state_animation(self, head_idx: int = 0, fps: int = 5,
                               save: bool = True) -> Optional[str]:
        """
        创建状态矩阵演化的动画

        Args:
            head_idx: head 索引
            fps: 帧率

        Returns:
            保存的文件路径
        """
        if not self.hooks.state_history:
            print("No state history available")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        # 初始化
        state = self.hooks.state_history[0][0, head_idx].numpy()
        vmax = max(np.abs(s[0, head_idx]).max() for s in self.hooks.state_history)
        im = ax.imshow(state, aspect='auto', cmap='RdBu_r',
                      norm=Normalize(vmin=-vmax, vmax=vmax))
        ax.set_xlabel('d_v')
        ax.set_ylabel('d_k')
        title = ax.set_title(f'State Matrix (Head {head_idx}) - Step 0')
        plt.colorbar(im, ax=ax)

        def update(frame):
            state = self.hooks.state_history[frame][0, head_idx].numpy()
            im.set_array(state)
            title.set_text(f'State Matrix (Head {head_idx}) - Step {frame}')
            return [im, title]

        anim = animation.FuncAnimation(fig, update, frames=len(self.hooks.state_history),
                                       interval=1000//fps, blit=True)

        if save:
            path = self.save_dir / f'state_evolution_head{head_idx}.gif'
            anim.save(path, writer='pillow', fps=fps)
            plt.close(fig)
            return str(path)
        else:
            plt.show()
            return None

    def plot_update_mask_pattern(self, step_idx: int = -1, save: bool = True,
                                  show: bool = False) -> Optional[plt.Figure]:
        """绘制更新掩码模式"""
        if not self.hooks.update_mask_history:
            print("No update mask history available")
            return None

        mask = self.hooks.update_mask_history[step_idx].numpy()  # [E, B, L, H]

        fig, axes = plt.subplots(1, mask.shape[1], figsize=(5*mask.shape[1], 4))
        if mask.shape[1] == 1:
            axes = [axes]

        for b, ax in enumerate(axes):
            # 对每个 batch，显示专家激活模式
            mask_b = mask[:, b, :, 0]  # [E, L] 取第一个 head
            im = ax.imshow(mask_b, aspect='auto', cmap='Greens', vmin=0, vmax=1)
            ax.set_yticks(range(len(self.expert_names)))
            ax.set_yticklabels(self.expert_names)
            ax.set_xlabel('Sequence Position')
            ax.set_title(f'Batch {b} Update Mask')
            plt.colorbar(im, ax=ax)

        plt.suptitle('Expert Update Mask Pattern', fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir / 'update_mask_pattern.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def generate_report(self, save: bool = True) -> Dict[str, Any]:
        """生成综合报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_steps': self.hooks.step_count,
            'statistics': {}
        }

        # 输出权重统计
        if self.hooks.output_weights_history:
            latest_weights = self.hooks.output_weights_history[-1].numpy()
            report['statistics']['output_weights'] = {
                'latest': latest_weights.tolist(),
                'per_expert_mean': latest_weights.mean(axis=1).tolist(),
            }

        # 损失统计
        if self.hooks.loss_history:
            report['statistics']['loss'] = {
                'latest': self.hooks.loss_history[-1],
                'mean': np.mean(self.hooks.loss_history),
                'min': np.min(self.hooks.loss_history),
            }

        # 梯度统计
        if self.hooks.grad_norm_history:
            report['statistics']['gradient_norm'] = {
                'latest': self.hooks.grad_norm_history[-1],
                'mean': np.mean(self.hooks.grad_norm_history),
                'max': np.max(self.hooks.grad_norm_history),
            }

        if save:
            with open(self.save_dir / 'report.json', 'w') as f:
                json.dump(report, f, indent=2)

        return report

    def plot_all(self, show: bool = False):
        """生成所有可用的可视化"""
        print("Generating visualizations...")

        self.plot_output_weights(save=True, show=show)
        self.plot_training_curves(save=True, show=show)
        self.plot_gradient_flow(save=True, show=show)
        self.plot_expert_contributions(save=True, show=show)
        self.plot_gate_distribution(save=True, show=show)
        self.plot_gate_heatmap(save=True, show=show)
        self.plot_update_mask_pattern(save=True, show=show)
        self.plot_state_evolution(save=True, show=show)
        self.generate_report(save=True)

        print(f"All visualizations saved to {self.save_dir}")


def create_visualization_callback(hooks: MultimodalVisualizationHooks,
                                   visualizer: MultimodalVisualizer,
                                   plot_every_n_steps: int = 100):
    """
    创建训练回调函数

    用法:
        callback = create_visualization_callback(hooks, visualizer, plot_every_n_steps=100)

        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            callback(step, loss, model)
    """
    def callback(step: int, loss: float, model: nn.Module):
        hooks.record_loss(loss)
        hooks.record_grad_norm(model)
        hooks.step()

        if step > 0 and step % plot_every_n_steps == 0:
            visualizer.plot_all(show=False)

    return callback
