# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_multimodal_gated_deltanet import MultimodalGatedDeltaNetConfig
from .modeling_multimodal_gated_deltanet import (
    MultimodalGatedDeltaNetForCausalLM,
    MultimodalGatedDeltaNetModel
)

MODEL_TYPE = MultimodalGatedDeltaNetConfig.model_type
AutoConfig.register(MODEL_TYPE, MultimodalGatedDeltaNetConfig)
AutoModel.register(MultimodalGatedDeltaNetConfig, MultimodalGatedDeltaNetModel)
AutoModelForCausalLM.register(MultimodalGatedDeltaNetConfig, MultimodalGatedDeltaNetForCausalLM)

__all__ = [
    'MultimodalGatedDeltaNetConfig',
    'MultimodalGatedDeltaNetForCausalLM',
    'MultimodalGatedDeltaNetModel',
    'MODEL_TYPE'
]
