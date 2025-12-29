# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mob_gated_deltanet_moe import \
    MobGatedDeltaNetMoEConfig
from .modeling_mob_gated_deltanet_moe import (
    MobGatedDeltaNetMoEForCausalLM, MobGatedDeltaNetMoEModel)

MODEL_TYPE = MobGatedDeltaNetMoEConfig.model_type
AutoConfig.register(MODEL_TYPE, MobGatedDeltaNetMoEConfig)
AutoModel.register(MobGatedDeltaNetMoEConfig, MobGatedDeltaNetMoEModel)
AutoModelForCausalLM.register(MobGatedDeltaNetMoEConfig, MobGatedDeltaNetMoEForCausalLM)

__all__ = ['MobGatedDeltaNetMoEConfig', 'MobGatedDeltaNetMoEForCausalLM', 'MobGatedDeltaNetMoEModel', 'MODEL_TYPE']
