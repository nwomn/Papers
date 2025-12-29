# -*- coding: utf-8 -*-
"""
Profiling script for MobGatedDeltaNetMoE layer
分析前向传播和反向传播的性能瓶颈

独立版本：不依赖完整 layer.py 导入，只测试关键操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MockMobGatedDeltaNetMoE(nn.Module):
    """
    简化版 MobGatedDeltaNetMoE，只包含需要 profiling 的核心操作
    不包含 fla 依赖
    """
    def __init__(
        self,
        hidden_size=2048,
        expand_v=2,
        head_dim=256,
        num_heads=8,
        ratio=6,
        shared_head=2,
        topk=2,
        num_block=1,
        overlap=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.ratio = ratio
        self.shared_head = shared_head
        self.topk = topk
        self.num_block = num_block
        self.overlap = overlap

        self.key_dim = num_heads * head_dim
        self.value_dim = self.key_dim * expand_v
        self.head_qk_dim = head_dim
        self.head_v_dim = head_dim * expand_v

        # 核心投影层
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Per-head expert projections (Priority 1 优化目标)
        self.k_proj_expand = nn.ModuleList([
            nn.Linear(self.head_qk_dim, self.key_dim // self.num_heads * self.ratio, bias=False)
            for _ in range(self.num_heads)
        ])
        self.q_proj_expand = nn.ModuleList([
            nn.Linear(self.head_qk_dim, self.key_dim // self.num_heads * self.ratio, bias=False)
            for _ in range(self.num_heads)
        ])

        # Gating (Priority 2 优化目标)
        self.gate = nn.Linear(self.key_dim // self.num_heads, self.ratio - self.shared_head, bias=False)

        # Beta/G projections (Priority 3 优化目标)
        self.b_proj = nn.Linear(hidden_size, self.num_heads * self.ratio, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads * self.ratio, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.num_heads * self.ratio).uniform_(0, 16).log())
        self.dt_bias = nn.Parameter(torch.randn(self.num_heads * self.ratio))

        # Output
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def sparse(self, hidden_states):
        """Sparse routing (Priority 2)"""
        hidden_states = rearrange(hidden_states, 'h b l d -> (h b) l d', h=self.num_heads).contiguous()
        router_logits = self.gate(hidden_states)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        selected_memories = selected_memories + self.shared_head
        routing_weights = routing_weights.to(hidden_states.dtype)
        router_weight_full = torch.zeros(
            (routing_weights.shape[0], routing_weights.shape[1], self.ratio),
            dtype=routing_weights.dtype, device=routing_weights.device
        ).scatter(-1, selected_memories, routing_weights)
        if self.shared_head > 0:
            router_weight_full[:, :, 0:self.shared_head] = 1 / self.shared_head
        router_weight_full = router_weight_full / router_weight_full.sum(dim=-1, keepdim=True)
        router_weight_full = rearrange(router_weight_full, '(h b) l n -> n b l h', h=self.num_heads).contiguous()
        router_mask = router_weight_full.bool().int()
        return router_mask, router_weight_full

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # Initial projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k, v = map(lambda x: rearrange(x, 'b t (h d) -> h b t d', h=self.num_heads), (q, k, v))

        # Priority 2: Sparse routing
        router_mask, router_weight_full = self.sparse(q)

        # Priority 1: Per-head expert projections (THE BOTTLENECK)
        k = torch.stack([k_expert(k[i]) for i, k_expert in enumerate(self.k_proj_expand)], dim=0)
        q = torch.stack([q_expert(q[i]) for i, q_expert in enumerate(self.q_proj_expand)], dim=0)

        # Reshape
        k, q = (rearrange(x, 'h b l (e d) -> e h b l d', e=self.ratio) for x in (k, q))
        v = repeat(v, 'h b l d -> e h b l d', e=self.ratio).contiguous()
        k, v, q = (rearrange(x, 'e h b l d -> (e b) l (h d)', e=self.ratio) for x in (k, v, q))

        # Reshape back for mock computation
        k, v, q = (rearrange(x, '(e b) l (h d) -> e b l h d', e=self.ratio, h=self.num_heads) for x in (k, v, q))
        k = k * router_mask[..., None]
        v = v * router_mask[..., None]
        q = q * router_mask[..., None]

        # Priority 3: Gating computation
        beta = rearrange(self.b_proj(hidden_states).sigmoid(), 'b l (e h) -> e b l h', e=self.ratio, h=self.num_heads) * router_mask
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
        g = rearrange(g, 'b l (e h) -> e b l h', e=self.ratio, h=self.num_heads) * router_mask

        # Mock attention computation (代替 chunk_gated_delta_rule)
        # 简单用 scaled dot-product attention 模拟计算量
        q_attn = rearrange(q, 'e b l h d -> (e b) h l d')
        k_attn = rearrange(k, 'e b l h d -> (e b) h l d')
        v_attn = rearrange(v, 'e b l h d -> (e b) h l d')

        # Scaled dot-product attention
        attn_weights = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        o = torch.matmul(attn_weights, v_attn)

        o = rearrange(o, '(e b) h l d -> e b l h d', e=self.ratio)

        # Priority 4: Branch aggregation
        o = torch.einsum('nblhd,nblh->blhd', o.to(router_weight_full.dtype), router_weight_full)

        # Output projection
        g_out = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        o = o * F.silu(g_out)  # 简化版 gating
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o


def time_op(name, op_fn, n_runs=100, warmup=10):
    """精确计时单个操作"""
    # 预热
    for _ in range(warmup):
        op_fn()
    torch.cuda.synchronize()

    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_runs):
        op_fn()
    end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / n_runs
    print(f"  {name}: {avg_time:.4f} ms")
    return avg_time


def profile_individual_ops(batch_size=4, seq_len=512, hidden_size=2048):
    """
    单独测试各个操作的耗时
    """
    print("\n" + "=" * 60)
    print("Individual Operation Profiling")
    print("=" * 60)
    print(f"  batch_size:   {batch_size}")
    print(f"  seq_len:      {seq_len}")
    print(f"  hidden_size:  {hidden_size}")
    print("=" * 60)

    device = 'cuda'
    dtype = torch.bfloat16

    layer = MockMobGatedDeltaNetMoE(hidden_size=hidden_size).to(device).to(dtype)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    num_heads = layer.num_heads
    head_dim = layer.head_dim
    ratio = layer.ratio

    # 准备各阶段的输入
    q_projected = layer.q_proj(x)
    k_projected = layer.k_proj(x)
    v_projected = layer.v_proj(x)

    q = rearrange(q_projected, 'b t (h d) -> h b t d', h=num_heads)
    k = rearrange(k_projected, 'b t (h d) -> h b t d', h=num_heads)

    results = {}

    # =========================================
    # Priority 1: Per-head Expert Projections
    # =========================================
    print("\n[Priority 1] Per-Head Expert Projections:")

    def expert_proj_loop():
        k_out = torch.stack([k_expert(k[i]) for i, k_expert in enumerate(layer.k_proj_expand)], dim=0)
        q_out = torch.stack([q_expert(q[i]) for i, q_expert in enumerate(layer.q_proj_expand)], dim=0)
        return k_out, q_out

    results['expert_proj'] = time_op("k/q expert projection (Python loop)", expert_proj_loop)

    # 对比：如果用单个大 Linear 会怎样
    combined_weight_k = torch.stack([m.weight for m in layer.k_proj_expand], dim=0)  # (H, out, in)
    combined_weight_q = torch.stack([m.weight for m in layer.q_proj_expand], dim=0)

    def expert_proj_bmm():
        # k: (H, B, L, D) -> (H, B*L, D)
        k_flat = k.reshape(num_heads, -1, head_dim)
        q_flat = q.reshape(num_heads, -1, head_dim)
        # bmm: (H, B*L, D) @ (H, D, out) -> (H, B*L, out)
        k_out = torch.bmm(k_flat, combined_weight_k.transpose(-1, -2))
        q_out = torch.bmm(q_flat, combined_weight_q.transpose(-1, -2))
        return k_out, q_out

    results['expert_proj_bmm'] = time_op("k/q expert projection (batched bmm)", expert_proj_bmm)

    # =========================================
    # Priority 2: Sparse Routing
    # =========================================
    print("\n[Priority 2] Sparse Routing:")

    def sparse_routing():
        return layer.sparse(q)

    results['sparse'] = time_op("sparse routing (full)", sparse_routing)

    # 分解 sparse routing 的各个步骤
    hidden_states_sparse = rearrange(q, 'h b l d -> (h b) l d', h=num_heads).contiguous()

    def sparse_gate():
        return layer.gate(hidden_states_sparse)
    results['sparse_gate'] = time_op("  - gate linear", sparse_gate)

    router_logits = layer.gate(hidden_states_sparse)

    def sparse_softmax():
        return F.softmax(router_logits, dim=2, dtype=torch.float)
    results['sparse_softmax'] = time_op("  - softmax", sparse_softmax)

    scores = F.softmax(router_logits, dim=2, dtype=torch.float)

    def sparse_topk():
        return torch.topk(scores, layer.topk, dim=-1)
    results['sparse_topk'] = time_op("  - topk", sparse_topk)

    routing_weights, selected_memories = torch.topk(scores, layer.topk, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    selected_memories = selected_memories + layer.shared_head
    routing_weights = routing_weights.to(dtype)

    def sparse_scatter():
        return torch.zeros(
            (routing_weights.shape[0], routing_weights.shape[1], ratio),
            dtype=routing_weights.dtype, device=routing_weights.device
        ).scatter(-1, selected_memories, routing_weights)
    results['sparse_scatter'] = time_op("  - scatter", sparse_scatter)

    # =========================================
    # Priority 3: Gating Computation
    # =========================================
    print("\n[Priority 3] Gating Computation:")

    router_mask, router_weight_full = layer.sparse(q)

    def gating_beta():
        return rearrange(layer.b_proj(x).sigmoid(), 'b l (e h) -> e b l h', e=ratio, h=num_heads) * router_mask
    results['gating_beta'] = time_op("beta computation", gating_beta)

    def gating_g():
        g = -layer.A_log.float().exp() * F.softplus(layer.a_proj(x).float() + layer.dt_bias)
        return rearrange(g, 'b l (e h) -> e b l h', e=ratio, h=num_heads) * router_mask
    results['gating_g'] = time_op("g computation", gating_g)

    # =========================================
    # Priority 4: Branch Aggregation
    # =========================================
    print("\n[Priority 4] Branch Aggregation:")

    # 创建模拟输出
    o_mock = torch.randn(ratio, batch_size, seq_len, num_heads, head_dim * 2, device=device, dtype=dtype)

    def branch_agg():
        return torch.einsum('nblhd,nblh->blhd', o_mock, router_weight_full)
    results['branch_agg'] = time_op("einsum aggregation", branch_agg)

    # =========================================
    # Full Forward/Backward
    # =========================================
    print("\n[Full Forward/Backward]:")

    def full_forward():
        return layer(x)
    results['full_forward'] = time_op("complete forward", full_forward, n_runs=50)

    def forward_backward():
        x_grad = x.detach().clone().requires_grad_(True)
        out = layer(x_grad)
        loss = out.sum()
        loss.backward()
    results['forward_backward'] = time_op("forward + backward", forward_backward, n_runs=30)

    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_forward = results['full_forward']

    print(f"\nTotal forward time: {total_forward:.4f} ms\n")

    print("Breakdown of identified bottlenecks:")
    print("-" * 50)

    bottlenecks = [
        ('Priority 1: Expert Projection (loop)', results['expert_proj']),
        ('Priority 1: Expert Projection (bmm)', results['expert_proj_bmm']),
        ('Priority 2: Sparse Routing', results['sparse']),
        ('Priority 3: Gating (beta)', results['gating_beta']),
        ('Priority 3: Gating (g)', results['gating_g']),
        ('Priority 4: Branch Aggregation', results['branch_agg']),
    ]

    for name, time_ms in bottlenecks:
        pct = time_ms / total_forward * 100
        bar = '█' * int(pct / 2)
        print(f"  {name:40s} {time_ms:8.4f} ms ({pct:5.1f}%) {bar}")

    print("-" * 50)
    print(f"\nForward + Backward: {results['forward_backward']:.4f} ms")
    print(f"Backward overhead: {results['forward_backward'] - total_forward:.4f} ms")

    # Speedup potential
    print("\n" + "=" * 60)
    print("OPTIMIZATION POTENTIAL")
    print("=" * 60)
    speedup = results['expert_proj'] / results['expert_proj_bmm']
    print(f"\nExpert Projection: Loop vs BMM")
    print(f"  Loop:  {results['expert_proj']:.4f} ms")
    print(f"  BMM:   {results['expert_proj_bmm']:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile MobGatedDeltaNetMoE layer")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=2048, help="Hidden size")

    args = parser.parse_args()

    profile_individual_ops(
        batch_size=args.batch,
        seq_len=args.seq,
        hidden_size=args.hidden,
    )
