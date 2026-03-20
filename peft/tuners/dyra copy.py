# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
    Linear
)

from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class DyraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_CoIN`]
    """
    lora_init_file: str = field(default="")

    def __post_init__(self):
        self.peft_type = PeftType.DYRA


class DyraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config

        self.lora_inits = {}

        self.add_adapter(adapter_name, self.peft_config[adapter_name])
    
    def add_adapter(self, adapter_name, config=None):
        if config.lora_init_file:
            self.lora_inits[adapter_name] = torch.load(config.lora_init_file)
        if config is not None:
            model_config = getattr(self.model, "config", {"model_type": "custom"})
            if hasattr(model_config, "to_dict"):
                model_config = model_config.to_dict()

            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target, key)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target, key):
        lora_r, lora_A, lora_B = self._get_lora_init_for_tgt_layer_and_task(adapter_name, key, 0)
        if lora_r is None:
            lora_r = lora_config.r

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": False,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        
        # if loaded_in_4bit:
        #     print(f"Layer: {key}, adapter_name: {adapter_name} loaded_in_4bit")
        # if loaded_in_8bit:
        #     print(f"Layer: {key}, adapter_name: {adapter_name} loaded_in_8bit")

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
            if lora_A is not None and lora_B is not None:
                new_module.lora_A[adapter_name].weight.data.copy_(lora_A.T)
                new_module.lora_B[adapter_name].weight.data.copy_(lora_B)
            else:
                warnings.warn(f'Failed find init for layer: {key}')
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
            if lora_A is not None and lora_B is not None:
                new_module.lora_A[adapter_name].weight.data.copy_(lora_A.T)
                new_module.lora_B[adapter_name].weight.data.copy_(lora_B)
            else:
                warnings.warn(f'Failed find init for layer: {key}')
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
            if lora_A is not None and lora_B is not None:
                new_module.lora_A[adapter_name].weight.data.copy_(lora_A.T)
                new_module.lora_B[adapter_name].weight.data.copy_(lora_B)
            else:
                warnings.warn(f'Failed find init for layer: {key}')
        return new_module
    
    def _get_lora_init_for_tgt_layer_and_task(self, adapter_name, key, task_id=0):
        lora_init = self.lora_inits.get(adapter_name, {}).get(task_id, {})
        r, lora_A, lora_B = None, None, None
        for k, v in lora_init.items():
            if key in k and 'lora_A' in k:
                lora_A = v
                r = min(lora_A.shape)
            elif key in k and 'lora_B' in k:
                lora_B = v
                r = min(lora_B.shape)
        return r, lora_A, lora_B
    