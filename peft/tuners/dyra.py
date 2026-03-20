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

def mark_only_cur_lora_as_trainable(model: nn.Module, adapter_name: str = "default", bias: str = "none", cur_task: int = 0) -> None:
    for n, p in model.named_parameters():
        if f"lora_A.{adapter_name}.{cur_task}." in n or f"lora_B.{adapter_name}.{cur_task}." in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


@dataclass
class MoedyraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_CoIN`]
    """
    lora_init_file: str = field(default="")
    num_experts: int = field(default=8)
    cur_task: int = field(default=0)

    def __post_init__(self):
        self.peft_type = PeftType.MOEDYRA


class MoedyraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config

        self.lora_inits = {}

        self.add_adapter(adapter_name, self.peft_config[adapter_name])
    
    def add_adapter(self, adapter_name, config=None):
        self.cur_task = config.cur_task

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
        # mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        # mark_only_cur_lora_as_trainable(self.model, adapter_name, self.peft_config[adapter_name].bias, self.cur_task)
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
        lora_ranks = {}
        lora_A_inits = {}
        lora_B_inits = {}
        for task_id in range(lora_config.num_experts):
            lora_r, lora_A, lora_B = self._get_lora_init_for_tgt_layer_and_task(adapter_name, key, task_id)
            if (lora_r is None or lora_A is None or lora_B is None) and (task_id <= self.cur_task):
                warnings.warn(f"Failed to found lora_init! layer: {key}, task: {task_id}")

            lora_ranks[task_id] = lora_r if lora_r is not None else lora_config.r
            lora_A_inits[task_id] = lora_A
            lora_B_inits[task_id] = lora_B

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_ranks,
            "lora_A_inits": lora_A_inits,
            "lora_B_inits": lora_B_inits,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": False,
            "num_experts": lora_config.num_experts
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
            new_module = MoeDyraLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)
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
    
    def _set_expert_weights(self, expert_weights):
        for _, module in self.model.named_modules():
            if hasattr(module, '_set_expert_weights'):
                module._set_expert_weights(expert_weights)
    

class MoeDyraLinear(Linear):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        num_experts: int = 8,
        lora_A_inits: dict = {},
        lora_B_inits: dict = {},
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts, lora_A_inits, lora_B_inits)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

        self.expert_weights = torch.ones([num_experts]) / num_experts
    
    def _set_expert_weights(self, expert_weights):
        self.expert_weights = expert_weights

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts, lora_A_inits, lora_B_inits):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        lora_A_all = nn.ModuleList([])
        lora_B_all = nn.ModuleList([])
        self.scaling[adapter_name] = {}
        for expert_id in range(num_experts):
            expert_rank = r[expert_id]
            self.scaling[adapter_name][expert_id] = lora_alpha / expert_rank

            lora_A = nn.Linear(self.in_features, expert_rank, bias=False)
            lora_B = nn.Linear(expert_rank, self.out_features, bias=False)

            lora_A_init = lora_A_inits[expert_id]
            lora_B_init = lora_B_inits[expert_id]
            if lora_A_init is not None:
                lora_A.weight.data.copy_(lora_A_init.T)
            if lora_B_init is not None:
                lora_B.weight.data.copy_(lora_B_init)

            lora_A_all.append(lora_A)
            lora_B_all.append(lora_B)

        self.lora_A.update(nn.ModuleDict({adapter_name: lora_A_all}))
        self.lora_B.update(nn.ModuleDict({adapter_name: lora_B_all}))
            
        # if init_lora_weights:
        #     self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if any(v > 0 for v in self.r[self.active_adapter].values()) and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            dtype = self.lora_A[self.active_adapter][0].weight.dtype
            device = self.lora_A[self.active_adapter][0].weight.device
            x = x.to(dtype)
            expert_weights = self.expert_weights.to(dtype).to(device)
            active_experts = torch.nonzero(expert_weights).squeeze(-1)
            for i in active_experts.tolist():
                lora_A_i = self.lora_A[self.active_adapter][i]
                lora_B_i = self.lora_B[self.active_adapter][i]
                lora_dropout = self.lora_dropout[self.active_adapter]
                scaling = self.scaling[self.active_adapter][i]
                result += lora_B_i(lora_A_i(lora_dropout(x))) * scaling * expert_weights[i]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result