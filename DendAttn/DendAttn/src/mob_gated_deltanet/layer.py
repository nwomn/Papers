# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange,repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


@torch.compile
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


@torch.compile
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1

def re_process(x: torch.Tensor, n: int, overlap: int) -> torch.Tensor:
    """
    Split the last dimension (d) of the input tensor into n overlapping windows.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, num_heads, d)
        n: Number of windows to split into
        overlap: Number of overlapping elements between adjacent windows
    
    Returns:
        Output tensor of shape (batch_size, seq_len, num_heads * n, window_size)
    
    Raises:
        AssertionError: If parameters don't satisfy splitting condition
    """
    b, l, h, d = x.shape
    
    # Calculate window size and step
    # breakpoint()
    window_size = (d + (n - 1) * overlap) // n
    step = window_size - overlap
    
    # Validate parameters
    assert step > 0, "Step size must be positive"
    assert overlap >= 0, "Overlap must be non-negative"
    assert (n - 1) * step + window_size == d, (
        f"Parameters don't satisfy splitting condition: {(n - 1) * step + window_size} != {d}"
    )
    
    # Generate slices using list comprehension
    # breakpoint()
    slices = [x[..., i*step : i*step + window_size] for i in range(n)]
    
    # Concatenate along the head dimension
    # breakpoint()
    out = torch.cat(slices, dim=2)  # shape: (b, l, h*n, window_size)
    
    return out


class MobGatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.
    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional): ****deprecated****
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 8,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        ratio: int = 6, # add
        shared_head: int = 0, # add
        topk: int = 2, # add
        num_block: int = 1, # add
        overlap: int = 0, # add
        policy: str = 'token', # add
        **kwargs
    ) -> MobGatedDeltaNet:
        super().__init__()

        self.mode = mode

        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        
        self.ratio=ratio 
        self.shared_head=shared_head 
        self.topk = topk 
        self.num_block=num_block 
        self.overlap=overlap 
        self.policy=policy 
        
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.key_dim * self.expand_v
        self.head_qk_dim = head_dim
        self.head_v_dim = head_dim * self.expand_v
        self.layer_idx = layer_idx

        assert self.ratio >= self.shared_head + self.topk, "ratio must be larger than shared_head + topk"
        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."

        self.k_proj_expand =  nn.ModuleList([nn.Linear(self.head_qk_dim, self.key_dim//self.num_heads*self.ratio, bias=False) for _ in range(self.num_heads)])
        self.q_proj_expand = nn.ModuleList([nn.Linear(self.head_qk_dim, self.key_dim//self.num_heads*self.ratio, bias=False) for _ in range(self.num_heads)])
        self.gate = nn.Linear(self.key_dim//self.num_heads, self.ratio-self.shared_head, bias=False) if self.topk > 0 else None

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads*self.ratio, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads*self.ratio, bias=False)
        A = torch.empty(self.num_heads*self.ratio, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads*self.ratio) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
    #     self.apply(self._initialize_weights)
        
    # def _initialize_weights(self, module: nn.Module):
    #     if getattr(module, "_is_hf_initialized", False):
    #         return
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     module._is_hf_initialized = True

    def sparse(self,hidden_states,attention_mask):
        # breakpoint()
        # 1. 直接处理 topk=0：只保留 shared_head 的平均权重
        if self.topk == 0:
            B, L, H = hidden_states.shape[1], hidden_states.shape[2], self.num_heads
            assert self.shared_head > 0, "topk=0 时 shared_head 必须 > 0"
            router_weight_full = torch.zeros(
                (self.shared_head, B, L, H),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            router_weight_full.fill_(1.0 / self.shared_head)
            router_mask = torch.ones_like(router_weight_full, dtype=torch.int)
            return router_mask, router_weight_full, None
        
        # 2. 处理 topk > 0 的情况
        assert self.policy in ['token', 'mix', 'task'], f"Not suppoerted policy `{self.policy}`."
        hidden_states = rearrange(hidden_states, 'h b l d -> (h b) l d', h=self.num_heads).contiguous()
        hb, l, d = hidden_states.shape
        if self.policy == 'token':
            gate_input = hidden_states
        else:
            if l >= 3:
                start_idx = torch.argmax(attention_mask, dim=1) # (b,)
                start_idx = start_idx.repeat_interleave(self.num_heads) # (hb,)
                avg_token = (hidden_states[torch.arange(hb, device=hidden_states.device), start_idx + 1].add_(
                    hidden_states[torch.arange(hb, device=hidden_states.device), start_idx + 2]).div_(2))  # (hb, d) 
                gate_input = avg_token.unsqueeze(1).expand(-1, l, -1)  # (hb, l, d)
                self.avg_token = avg_token
            else:
                gate_input = self.avg_token.unsqueeze(1).expand(-1, l, -1)  # (hb, l, d)
            if self.policy == 'mix':
                gate_input = gate_input + hidden_states
        
        router_logits = self.gate(gate_input)  # (hb, l, ratio-shared_head)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # (hb, l, top_k)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # breakpoint()
        selected_memories = selected_memories+self.shared_head
        routing_weights = routing_weights.to(hidden_states.dtype)  # we cast back to the input dtype
        router_weight_full = torch.zeros((routing_weights.shape[0], routing_weights.shape[1], self.ratio), dtype=routing_weights.dtype, device=routing_weights.device).scatter(-1, selected_memories, routing_weights)
        if self.shared_head > 0: # 更改：添加以防止除零错误
            router_weight_full[:, :, 0:self.shared_head] = 1/self.shared_head
        router_weight_full = router_weight_full / router_weight_full.sum(dim=-1, keepdim=True) # 归一化路由权重 (原先是shared_head 1 + top_k 1 = 2)
        # breakpoint()
        router_weight_full = rearrange(router_weight_full, '(h b) l n -> n b l h', h=self.num_heads).contiguous()
        # breakpoint()
        router_mask = router_weight_full.bool().int()
        return router_mask,router_weight_full,router_logits


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k, v = map(lambda x: rearrange(x, 'b t (h d) -> h b t d', h=self.num_heads), (q, k, v))

        router_mask, router_weight_full, router_logits = self.sparse(q, attention_mask)
        
        k = torch.stack([k_expert(k[i]) for i, k_expert in enumerate(self.k_proj_expand)], dim=0)
        q = torch.stack([q_expert(q[i]) for i, q_expert in enumerate(self.q_proj_expand)], dim=0)
        # breakpoint()
        k,q= (rearrange(x, 'h b l (e d) -> e h b l d', e=self.ratio) for x in (k,q))
        v = repeat(v, 'h b l d -> e h b l d', e=self.ratio).contiguous()
        # breakpoint()
        k,v,q= (rearrange(x, 'e h b l d -> (e b) l (h d)', e=self.ratio) for x in (k,v,q))

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            # print(f"conv_mask:{conv_mask.shape}\nattention_mask:{attention_mask}\n{attention_mask.shape}")
            conv_mask = repeat(conv_mask,'b l -> (e b) l', e=self.ratio).contiguous() if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            try:
                q, conv_state_q = self.q_conv1d(x=q,
                                                mask=conv_mask,
                                                cache=conv_state_q,
                                                output_final_state=use_cache,
                                                seq_idx=position_ids)
            except:
                breakpoint()
            k, conv_state_k = self.k_conv1d(x=k,
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
            try:
                v, conv_state_v = self.v_conv1d(x=v,
                                                mask=conv_mask,
                                                cache=conv_state_v,
                                                output_final_state=use_cache,
                                                seq_idx=position_ids)
            except:
                breakpoint()
            # breakpoint()
            k,v,q= (rearrange(x, '(e b) l (h d) -> e b l h d', e=self.ratio,h=self.num_heads) for x in (k,v,q))
            k = k*router_mask[...,None] 
            v = v*router_mask[...,None] 
            q = q*router_mask[...,None] 
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
            # q = F.silu(self.q_proj(hidden_states))
            # k = F.silu(self.k_proj(hidden_states))
            # v = F.silu(self.v_proj(hidden_states))


        # v = torch.stack([v_expert(v[i]) for i, v_expert in enumerate(self.v_proj_expand)], dim=0)

        beta = rearrange(self.b_proj(hidden_states).sigmoid(), 'b l (e h) -> e b l h', e=self.ratio,h=self.num_heads)*router_mask
        # breakpoint()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
        g = rearrange(g, 'b l (e h) -> e b l h', e=self.ratio,h=self.num_heads)*router_mask
        
        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)
        overlap = self.overlap
        num_block= self.num_block
        if mode == 'chunk':
            #breakpoint()
            q,k,v,g,beta= (rearrange(x, 'e b l h ... -> b l (e h) ...', e=self.ratio) for x in (q,k,v,g,beta))

            if num_block > 1: # block为1时取消分block
                q, k = map(lambda x: re_process(x,n=num_block,overlap=overlap), (q, k)) 
                # breakpoint()
                v, g, beta = map(lambda x: repeat(x, 'b t h ... -> b t (k h) ...', k=num_block).contiguous(), (v, g, beta)) 
            # else:
            #     assert (q, k == map(lambda x: re_process(x,n=num_block,overlap=overlap), (q, k))), "not consistent with re_process"
            #     assert (v, g, beta == map(lambda x: repeat(x, 'b t h ... -> b t (k h) ...', k=num_block).contiguous(), (v, g, beta))), "not consistent with repeat"
            #     print("In chunk: consistent with re_process and repeat")

            o, recurrent_state = chunk_gated_delta_rule(
                q=q.to(torch.bfloat16),
                k=k.to(torch.bfloat16),
                v=v.to(torch.bfloat16),
                g=g.to(torch.bfloat16),
                beta=beta.to(torch.bfloat16),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True
            )
            # breakpoint()
            o = rearrange(o, 'b t (k h) d -> b t h k d', k=num_block)
            # print(o.shape)
            o = o.sum(-2)
            
            o = rearrange(o, 'b l (e h) ... -> e b l h ...', e=self.ratio)
            o = torch.einsum('nblhd,nblh->blhd', o.to(router_weight_full.dtype), router_weight_full) # type更正对齐
            # o = o.sum(0)
        elif mode == 'fused_recurrent':
            q,k,v,g,beta= (rearrange(x, 'e b l h ... -> b l (e h) ...', e=self.ratio) for x in (q,k,v,g,beta))

            # overlap = 64
            if num_block > 1: # block为1时取消分block
                q, k = map(lambda x: re_process(x,n=num_block,overlap=overlap), (q, k))
                # breakpoint()
                v, g, beta = map(lambda x: repeat(x, 'b t h ... -> b t (k h) ...', k=num_block).contiguous(), (v, g, beta)) 
            # else:
            #     assert (q, k == map(lambda x: re_process(x,n=num_block,overlap=overlap), (q, k))), "not consistent with re_process"
            #     assert (v, g, beta == map(lambda x: repeat(x, 'b t h ... -> b t (k h) ...', k=num_block).contiguous(), (v, g, beta))), "not consistent with repeat"
            #     print("In fused_recurrent: consistent with re_process and repeat")
                
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                # head_first=False,  # 更正：适应新版本fla
                use_qk_l2norm_in_kernel=True
            )
            o = rearrange(o, 'b t (k h) d -> b t h k d', k=num_block)
            # print(o.shape)
            o = o.sum(-2)
            
            o = rearrange(o, 'b l (e h) ... -> e b l h ...', e=self.ratio)
            o = torch.einsum('nblhd,nblh->blhd', o.to(router_weight_full.dtype), router_weight_full) #type更正对齐
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        if self.use_gate:
            # breakpoint()
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        # breakpoint()
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values, router_logits

