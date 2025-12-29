# -*- coding: utf-8 -*-

from typing import Dict, List, Optional

from transformers.configuration_utils import PretrainedConfig


class MultimodalGatedDeltaNetConfig(PretrainedConfig):
    """
    Configuration for Multimodal Gated DeltaNet with separate expert states
    for different modalities (shared, text, vision).
    """
    model_type = 'multimodal_gated_deltanet'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        # Base model parameters
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: int = 2,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        head_dim: int = 256,
        num_heads: int = 6,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 21,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,

        # Multimodal expert parameters
        num_modalities: int = 2,                    # Number of modalities (text=0, vision=1)
        modality_names: List[str] = None,           # Modality names list

        # State space parameters
        num_block: int = 1,                         # Number of blocks
        overlap: int = 0,                           # Block overlap

        # Interleaved multimodal settings
        interleaved_mode: bool = False,             # Enable token-level modality routing
        image_token_id: int = None,                 # Image placeholder token ID (e.g., 32000)

        # Future extension: intra-group sparse routing
        use_intra_group_routing: bool = False,      # Enable intra-group routing
        experts_per_modality: int = 1,              # Experts per modality group
        intra_group_topk: int = 1,                  # Intra-group topk

        **kwargs
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.vocab_size = vocab_size

        # Multimodal expert parameters
        self.num_modalities = num_modalities
        self.modality_names = modality_names or ['text', 'vision']

        # State space parameters
        self.num_block = num_block
        self.overlap = overlap

        # Interleaved multimodal settings
        self.interleaved_mode = interleaved_mode
        self.image_token_id = image_token_id
        # Store token IDs as instance attributes for Layer access
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Intra-group routing extension
        self.use_intra_group_routing = use_intra_group_routing
        self.experts_per_modality = experts_per_modality
        self.intra_group_topk = intra_group_topk

        # Computed: total experts = 1 (shared) + num_modalities * experts_per_modality
        self.num_experts = 1 + num_modalities * experts_per_modality

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
