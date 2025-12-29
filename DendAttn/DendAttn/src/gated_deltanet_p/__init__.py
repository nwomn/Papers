# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_gated_deltanet_p import \
    GatedDeltaNetpConfig
from .modeling_gated_deltanet_p import (
    GatedDeltaNetpForCausalLM, GatedDeltaNetpModel)

MODEL_TYPE = GatedDeltaNetpConfig.model_type
AutoConfig.register(MODEL_TYPE, GatedDeltaNetpConfig)
AutoModel.register(GatedDeltaNetpConfig, GatedDeltaNetpModel)
AutoModelForCausalLM.register(GatedDeltaNetpConfig, GatedDeltaNetpForCausalLM)

__all__ = ['GatedDeltaNetpConfig', 'GatedDeltaNetpForCausalLM', 'GatedDeltaNetpModel', 'MODEL_TYPE']
