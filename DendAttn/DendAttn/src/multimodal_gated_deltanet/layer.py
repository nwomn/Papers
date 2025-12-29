# -*- coding: utf-8 -*-
# Multimodal Gated DeltaNet Layer Implementation
# Three expert branches: shared, text, vision
# State update controlled by modality_ids
# Output aggregation with learnable weights

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


# Modality constants
MODALITY_TEXT = 0
MODALITY_VISION = 1
MODALITY_SHARED = -1  # For special tokens (bos, eos, image markers, etc.)


def re_process(x: torch.Tensor, n: int, overlap: int) -> torch.Tensor:
    """
    Split the last dimension (d) of the input tensor into n overlapping windows.

    Args:
        x: Input tensor of shape (batch_size, seq_len, num_heads, d)
        n: Number of windows to split into
        overlap: Number of overlapping elements between adjacent windows

    Returns:
        Output tensor of shape (batch_size, seq_len, num_heads * n, window_size)
    """
    b, l, h, d = x.shape

    window_size = (d + (n - 1) * overlap) // n
    step = window_size - overlap

    assert step > 0, "Step size must be positive"
    assert overlap >= 0, "Overlap must be non-negative"
    assert (n - 1) * step + window_size == d, (
        f"Parameters don't satisfy splitting condition: {(n - 1) * step + window_size} != {d}"
    )

    slices = [x[..., i*step : i*step + window_size] for i in range(n)]
    out = torch.cat(slices, dim=2)

    return out


class MultimodalGatedDeltaNet(nn.Module):
    """
    Multimodal Gated DeltaNet Layer Implementation.

    Core features:
    1. Three fixed expert branches: shared, text, vision
    2. Three independent state spaces: S_shared, S_text, S_vision
    3. Selective state update based on modality_ids
    4. Output aggregation: o = w_1*(q*S_shared) + w_2*(q*S_text) + w_3*(q*S_vision)
       with learnable weights

    Args:
        hidden_size: Hidden dimension size
        expand_v: Value expansion ratio
        head_dim: Dimension per head
        num_heads: Number of attention heads
        mode: Computation mode ('chunk' or 'fused_recurrent')
        use_gate: Whether to use output gate
        use_short_conv: Whether to use short convolutions
        conv_size: Convolution kernel size
        layer_idx: Layer index
        num_modalities: Number of modalities (default 2: text, vision)
        num_block: Number of blocks for processing
        overlap: Block overlap
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
        num_modalities: int = 2,
        num_block: int = 1,
        overlap: int = 0,
        # Interleaved mode settings
        interleaved_mode: bool = False,
        image_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
        pad_token_id: int = None,
        # Reserved for future extension
        use_intra_group_routing: bool = False,
        experts_per_modality: int = 1,
        intra_group_topk: int = 1,
        **kwargs
    ) -> MultimodalGatedDeltaNet:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_v_dim = int(head_dim * self.expand_v)
        self.layer_idx = layer_idx

        self.num_modalities = num_modalities
        self.num_block = num_block
        self.overlap = overlap

        # Interleaved mode settings
        self.interleaved_mode = interleaved_mode
        self.image_token_id = image_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Extension parameters
        self.use_intra_group_routing = use_intra_group_routing
        self.experts_per_modality = experts_per_modality
        self.intra_group_topk = intra_group_topk

        # Total experts: 1 (shared) + num_modalities = 3
        self.num_experts = 1 + num_modalities

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        # ==================== Shared Projections ====================
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # ==================== Expert-specific K/Q Expand Projections ====================
        # Index: 0=shared, 1=text, 2=vision
        self.expert_k_proj_expand = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.head_dim, self.head_dim, bias=False)
                for _ in range(self.num_heads)
            ])
            for _ in range(self.num_experts)
        ])

        # Note: Q is NOT expanded per-expert
        # Same Q is used to query all expert states (like asking the same question to different memory banks)
        # Only K is expanded per-expert (different encoding for different modalities)

        # ==================== Beta and Gate Parameters ====================
        self.b_proj = nn.Linear(hidden_size, self.num_heads * self.num_experts, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads * self.num_experts, bias=False)

        # A_log and dt_bias initialization
        A = torch.empty(self.num_heads * self.num_experts, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads * self.num_experts) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ==================== Learnable Output Weights ====================
        # Shape: [num_experts, num_heads] - learnable weights for output aggregation
        self.output_weights = nn.Parameter(torch.zeros(self.num_experts, self.num_heads))

        # ==================== Short Convolution ====================
        if use_short_conv:
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
                "Do not turn it off unless you know what you are doing."
            )

        # ==================== Output Layers ====================
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _infer_modality_ids(
        self,
        input_ids: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Infer modality IDs from input_ids based on special token positions.

        Args:
            input_ids: [B, L] - input token IDs
            batch_size: B
            seq_len: L
            device: target device

        Returns:
            modality_ids: [B, L] - inferred modality IDs
                - MODALITY_TEXT (0): text tokens
                - MODALITY_VISION (1): image tokens
                - MODALITY_SHARED (-1): special tokens (bos, eos, pad, etc.)
        """
        # Default to text
        modality_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        # Mark image tokens as vision modality
        if self.image_token_id is not None:
            modality_ids[input_ids == self.image_token_id] = MODALITY_VISION

        # Mark special tokens as shared modality
        special_token_ids = []
        if self.bos_token_id is not None:
            special_token_ids.append(self.bos_token_id)
        if self.eos_token_id is not None:
            special_token_ids.append(self.eos_token_id)
        if self.pad_token_id is not None:
            special_token_ids.append(self.pad_token_id)

        for token_id in special_token_ids:
            modality_ids[input_ids == token_id] = MODALITY_SHARED

        return modality_ids

    def _get_update_mask(
        self,
        modality_ids: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Generate state update mask based on modality IDs.

        Supports both sequence-level and token-level modality IDs:
        - [B]: sequence-level (backward compatible)
        - [B, L]: token-level (interleaved mode)

        Args:
            modality_ids: [B] or [B, L] - modality IDs
            batch_size: B
            seq_len: L
            device, dtype

        Returns:
            update_mask: [E, B, L, H] - update mask for each expert
                - expert 0 (shared): always 1
                - expert 1 (text): 1 when modality_id == MODALITY_TEXT (0)
                - expert 2 (vision): 1 when modality_id == MODALITY_VISION (1)
                - MODALITY_SHARED (-1): only expert 0 is active
        """
        # Handle both [B] and [B, L] formats
        if modality_ids.dim() == 1:
            # Sequence-level: [B] -> expand to [B, L]
            modality_ids = modality_ids[:, None].expand(batch_size, seq_len)

        # Initialize mask: [E, B, L]
        update_mask = torch.zeros(
            self.num_experts, batch_size, seq_len,
            device=device, dtype=dtype
        )

        # Expert 0 (Shared): always active
        update_mask[0, :, :] = 1.0

        # Expert 1 (Text): modality_id == MODALITY_TEXT (0)
        update_mask[1, :, :] = (modality_ids == MODALITY_TEXT).to(dtype)

        # Expert 2 (Vision): modality_id == MODALITY_VISION (1)
        update_mask[2, :, :] = (modality_ids == MODALITY_VISION).to(dtype)

        # Note: MODALITY_SHARED (-1) positions only have expert 0 active,
        # which is the intended behavior for special tokens

        # Expand to head dimension: [E, B, L] -> [E, B, L, H]
        update_mask = update_mask[..., None].expand(
            self.num_experts, batch_size, seq_len, self.num_heads
        ).contiguous()

        return update_mask

    def _get_output_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get normalized output aggregation weights.

        Returns:
            weights: [E, H] - normalized weights for each expert per head
        """
        # Apply softmax to get normalized weights
        weights = F.softmax(self.output_weights, dim=0)  # [E, H]
        return weights.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        modality_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        vis_hooks: Optional[Any] = None,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states: [B, L, D] - input hidden states
            attention_mask: [B, L] - attention mask
            modality_ids: [B] or [B, L] - modality IDs
                - [B]: sequence-level (backward compatible)
                - [B, L]: token-level (interleaved mode)
                - If None, will be inferred from input_ids or default to all text
            input_ids: [B, L] - input token IDs (used for auto-inferring modality_ids)
            vis_hooks: Optional visualization hooks for capturing internal states

        Returns:
            output: [B, L, D]
            attentions: None
            past_key_values: Cache
            router_logits: None (no sparse routing)
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len]"
            )

        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # ==================== Modality ID Handling ====================
        if modality_ids is None:
            if input_ids is not None and self.interleaved_mode:
                # Auto-infer from input_ids in interleaved mode
                modality_ids = self._infer_modality_ids(
                    input_ids, batch_size, seq_len, device
                )
            else:
                # Default: all text (sequence-level)
                modality_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Select inference mode
        mode = 'fused_recurrent' if seq_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        # ==================== Get Historical State ====================
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # ==================== Base Projections ====================
        q_base = self.q_proj(hidden_states)  # [B, L, key_dim]
        k_base = self.k_proj(hidden_states)  # [B, L, key_dim]
        v_base = self.v_proj(hidden_states)  # [B, L, value_dim]

        # Reshape: [B, L, H*D] -> [H, B, L, D]
        q_base = rearrange(q_base, 'b l (h d) -> h b l d', h=self.num_heads)
        k_base = rearrange(k_base, 'b l (h d) -> h b l d', h=self.num_heads)
        v_base = rearrange(v_base, 'b l (h d) -> h b l d', h=self.num_heads)

        # ==================== Update Mask ====================
        update_mask = self._get_update_mask(
            modality_ids, batch_size, seq_len, device, dtype
        )  # [E, B, L, H]

        # ==================== Expert-specific K Expansion ====================
        # Q is SHARED across all experts (same question to different memory banks)
        # K is EXPANDED per-expert (different encoding for different modalities)
        # V is SHARED across all experts (same content, different indexing)

        k_expanded = []
        for e in range(self.num_experts):
            k_e = torch.stack([
                self.expert_k_proj_expand[e][h](k_base[h])
                for h in range(self.num_heads)
            ], dim=0)  # [H, B, L, D]
            k_expanded.append(k_e)

        # Q: repeat for all experts (same Q queries all states)
        # Shape: [H, B, L, D] -> [E, H, B, L, D]
        q_all = repeat(q_base, 'h b l d -> e h b l d', e=self.num_experts).contiguous()

        # K: stack expert-specific projections
        # Shape: [E, H, B, L, D]
        k_all = torch.stack(k_expanded, dim=0)

        # V: repeat for all experts (same content)
        # Shape: [H, B, L, D_v] -> [E, H, B, L, D_v]
        v_all = repeat(v_base, 'h b l d -> e h b l d', e=self.num_experts).contiguous()

        # Reshape for conv: [E, H, B, L, D] -> [(E*B), L, (H*D)]
        q_flat = rearrange(q_all, 'e h b l d -> (e b) l (h d)')
        k_flat = rearrange(k_all, 'e h b l d -> (e b) l (h d)')
        v_flat = rearrange(v_all, 'e h b l d -> (e b) l (h d)')

        # ==================== Short Convolution ====================
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None and 'conv_state' in last_state:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            conv_mask = attention_mask[:, -seq_len:] if attention_mask is not None else None
            conv_mask = repeat(conv_mask, 'b l -> (e b) l', e=self.num_experts).contiguous() if conv_mask is not None else None
            position_ids = kwargs.get('position_ids', None)

            q_flat, conv_state_q = self.q_conv1d(
                x=q_flat, mask=conv_mask, cache=conv_state_q,
                output_final_state=use_cache, seq_idx=position_ids
            )
            k_flat, conv_state_k = self.k_conv1d(
                x=k_flat, mask=conv_mask, cache=conv_state_k,
                output_final_state=use_cache, seq_idx=position_ids
            )
            v_flat, conv_state_v = self.v_conv1d(
                x=v_flat, mask=conv_mask, cache=conv_state_v,
                output_final_state=use_cache, seq_idx=position_ids
            )

        # Reshape: [(E*B), L, (H*D)] -> [E, B, L, H, D]
        q_all = rearrange(q_flat, '(e b) l (h d) -> e b l h d', e=self.num_experts, h=self.num_heads)
        k_all = rearrange(k_flat, '(e b) l (h d) -> e b l h d', e=self.num_experts, h=self.num_heads)
        v_all = rearrange(v_flat, '(e b) l (h d) -> e b l h d', e=self.num_experts, h=self.num_heads)

        # ==================== Apply Update Mask to K, V (NOT Q) ====================
        # Q is NOT masked - all experts use the same q for querying all states
        # K, V are masked - controls which states get updated
        mask_expanded = update_mask[..., None]  # [E, B, L, H, 1]
        k_all = k_all * mask_expanded
        v_all = v_all * mask_expanded

        # ==================== Beta and Gate Calculation ====================
        beta = self.b_proj(hidden_states).sigmoid()  # [B, L, E*H]
        beta = rearrange(beta, 'b l (e h) -> e b l h', e=self.num_experts, h=self.num_heads)
        beta = beta * update_mask  # Apply update mask

        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )
        g = rearrange(g, 'b l (e h) -> e b l h', e=self.num_experts, h=self.num_heads)
        g = g * update_mask

        # Handle padding
        if attention_mask is not None:
            attn_mask = attention_mask[:, -seq_len:, None]
            beta = beta * attn_mask
            g = g * attn_mask

        # ==================== Visualization Hooks ====================
        if vis_hooks is not None and hasattr(vis_hooks, 'capture_gates'):
            vis_hooks.capture_gates(beta, g, update_mask)

        # ==================== Recurrent State ====================
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        # ==================== Gated Delta Rule ====================
        # Merge: [E, B, L, H, D] -> [B, L, (E*H), D]
        q_merged = rearrange(q_all, 'e b l h d -> b l (e h) d')
        k_merged = rearrange(k_all, 'e b l h d -> b l (e h) d')
        v_merged = rearrange(v_all, 'e b l h d -> b l (e h) d')
        g_merged = rearrange(g, 'e b l h -> b l (e h)')
        beta_merged = rearrange(beta, 'e b l h -> b l (e h)')

        num_block = self.num_block
        overlap = self.overlap

        if num_block > 1:
            q_merged, k_merged = map(
                lambda x: re_process(x, n=num_block, overlap=overlap),
                (q_merged, k_merged)
            )
            v_merged, g_merged, beta_merged = map(
                lambda x: repeat(x, 'b t h ... -> b t (k h) ...', k=num_block).contiguous(),
                (v_merged, g_merged, beta_merged)
            )

        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q_merged.to(torch.bfloat16),
                k=k_merged.to(torch.bfloat16),
                v=v_merged.to(torch.bfloat16),
                g=g_merged.to(torch.bfloat16),
                beta=beta_merged.to(torch.bfloat16),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True
            )
        else:  # fused_recurrent
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q_merged,
                k=k_merged,
                v=v_merged,
                g=g_merged,
                beta=beta_merged,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True
            )

        # Handle num_block > 1
        if num_block > 1:
            o = rearrange(o, 'b t (k h) d -> b t h k d', k=num_block)
            o = o.sum(-2)

        # Reshape back: [B, L, (E*H), D] -> [E, B, L, H, D]
        o = rearrange(o, 'b l (e h) d -> e b l h d', e=self.num_experts, h=self.num_heads)

        # ==================== Visualization Hooks (Expert Outputs) ====================
        if vis_hooks is not None and hasattr(vis_hooks, 'capture_expert_outputs'):
            vis_hooks.capture_expert_outputs(o)

        # ==================== Output Aggregation with Learnable Weights ====================
        # All three states participate in output (including non-updated ones)
        output_weights = self._get_output_weights(device, o.dtype)  # [E, H]
        o = torch.einsum('eblhd,eh->blhd', o, output_weights)

        # Convert back to original dtype
        o = o.to(dtype)

        # ==================== Update Cache ====================
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=seq_len
            )

        # ==================== Output Gate and Projection ====================
        if self.use_gate:
            g_out = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b l h d -> b l (h d)')
        o = o.to(dtype)  # Ensure correct dtype before final projection
        o = self.o_proj(o)

        return o, None, past_key_values, None
