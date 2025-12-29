# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mob_gated_deltanet import \
    MobGatedDeltaNetConfig
from .modeling_mob_gated_deltanet import (
    MobGatedDeltaNetForCausalLM, MobGatedDeltaNetModel)

MODEL_TYPE = MobGatedDeltaNetConfig.model_type
AutoConfig.register(MODEL_TYPE, MobGatedDeltaNetConfig)
AutoModel.register(MobGatedDeltaNetConfig, MobGatedDeltaNetModel)
AutoModelForCausalLM.register(MobGatedDeltaNetConfig, MobGatedDeltaNetForCausalLM)

__all__ = ['MobGatedDeltaNetConfig', 'MobGatedDeltaNetForCausalLM', 'MobGatedDeltaNetModel', 'MODEL_TYPE']
