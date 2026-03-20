import warnings
import math
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, field


from ..import_utils import is_bnb_4bit_available, is_bnb_available

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
)
from transformers.pytorch_utils import Conv1D

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class DyMoeLoraConfig(LoraConfig):
    expert_num: int = field(default=4)
    topk: int = field(default=2)
    lora_init_file: str = field(default="")
    cur_task: int = field(default=0)

    def __post_init__(self):
        self.peft_type = PeftType.DYMOELORA


class DyMoeLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name: str):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.moe_layers = {}
        self.lora_inits = {}

        self.add_adapter(adapter_name, self.peft_config[adapter_name])
    
    def set_expert_weights(self, expert_weights, adapter_name=None):
        if adapter_name is None:
            adapter_name = self.active_adapter

        for _, module in self.model.named_modules():
            if hasattr(module, 'set_expert_weights'):
                module.set_expert_weights(expert_weights)
    
    def add_adapter(self, adapter_name, config=None):
        if config.lora_init_file:
            self.lora_inits[adapter_name] = torch.load(config.lora_init_file)
        self.moe_layers[adapter_name] = []
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)   # load config
            self.peft_config[adapter_name] = config # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
    
    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue
            
            # if adapter_name in self.lora_ranks and not key in self.lora_ranks[adapter_name].keys():
            #     raise ValueError(f"Can not find lora rank for layer: {key}")

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
                self.moe_layers[adapter_name].append(new_module)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
    
    def _get_lora_init_for_tgt_layer_and_task(self, adapter_name, key, task_id):
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
    
    def _create_new_module(self, lora_config, adapter_name, target, key):
        all_ranks, all_lora_A, all_lora_B = {}, {}, {}
        for task_id in range(lora_config.cur_task+1):
            lora_r, lora_A, lora_B = self._get_lora_init_for_tgt_layer_and_task(adapter_name, key, task_id)
            if lora_r is None:
                warnings.warn(f"Cound not find lora init for {key}, task: {task_id}!")
                lora_r = lora_config.r

            all_ranks[task_id] = lora_r
            all_lora_A[task_id] = lora_A
            all_lora_B[task_id] = lora_B

        kwargs = {
            "r": all_ranks,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": False,
            "expert_num": lora_config.expert_num,
            "topk": lora_config.topk,
        }

        bias = hasattr(target, "bias") and target.bias is not None
    
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

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
            new_module = DyMoeLoraLinear(
                adapter_name, in_features, out_features, bias=bias, **kwargs)
            for task_id in range(lora_config.cur_task+1):
                lora_A = all_lora_A[task_id]
                lora_B = all_lora_B[task_id]
                if lora_A is None or lora_B is None:
                    warnings.warn(f"find lora_A or lora_B is None for {key}, task {task_id}")
                new_module.experts[adapter_name][task_id].lora_A.weight.data.copy_(lora_A.T)
                new_module.experts[adapter_name][task_id].lora_B.weight.data.copy_(lora_B)
        return new_module


class DyMoeLoraLayer:
    def __init__(self, in_features: int, out_features: int, expert_num: int, topk: int, **kwargs):
        self.experts = nn.ModuleDict({})

        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.expert_num = expert_num
        self.topk = topk

        self.kwargs = kwargs
        
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        lora_experts = nn.ModuleList([])
        for expert_id in range(self.expert_num):
            expert_rank = r.get(expert_id, 16)
            lora_experts.append(
                LoraExpert(in_features=self.in_features, out_features=self.out_features, r=expert_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights)
            )

        self.experts.update(
            nn.ModuleDict({adapter_name: lora_experts})
            )
        
        self.expert_weights = {adapter_name: torch.ones([1, self.expert_num], dtype=torch.bool) / self.expert_num}  # batch_size, expert_num
        self.topk = {adapter_name: self.topk}

        self.to(self.weight.device)
    
    def get_delta_weight(self, adapter, expert_id=0):
        lora = self.experts[adapter][expert_id]
        lora_A = lora.lora_A
        lora_B = lora.lora_B
        scaling = self.scaling
        return (
            transpose(
                lora_B.weight @ lora_A.weight,
                self.fan_in_fan_out,
            )
            * scaling
        )

class DyMoeLoraLinear(nn.Linear, DyMoeLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        expert_num: int = 8,
        topk: int = 2,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DyMoeLoraLayer.__init__(self, in_features=in_features, out_features=out_features, expert_num=expert_num, topk=topk)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
    
    def forward(self, x):
        if self.active_adapter not in self.experts.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:   # No adapter
            if self.merged: # merge the adapter to linear
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            return result
        elif not self.merged:
            base_output = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            dtype = self.experts[self.active_adapter][0].lora_A.weight.dtype
            x = x.to(dtype)     # B, L, d

            # expert_weights = self.expert_weights[self.active_adapter].to(dtype)           # (B, T)
            # topk = self.topk[self.active_adapter]
            # topk = min(topk, self.expert_num)

            # topk_values, topk_indices = torch.topk(expert_weights, k=topk, dim=-1)  # (B, topk), (B, topk)
            # lora_output = torch.zeros_like(base_output)
            # unique_experts = torch.unique(topk_indices)

            # for expert_idx in unique_experts:
            #     # 找到哪些样本选择了该专家，以及对应的权重
            #     sample_indices, pos_in_topk = torch.where(topk_indices == expert_idx)
            #     weights = topk_values[sample_indices, pos_in_topk].to(x.device)  # (num_samples_using_this_expert,)

            #     # 计算该专家的输出
            #     expert_out = self.experts[self.active_adapter][expert_idx](x)  # (B, ...)

            #     # 构建权重掩码，只对选中样本生效
            #     weight_mask = torch.zeros(x.size(0), device=x.device, dtype=expert_out.dtype)
            #     weight_mask[sample_indices] = weights

            #     # 将权重掩码扩展到与 expert_out 相同的维度（保留 batch 维度，其他维度视为1以便广播）
            #     extra_dims = expert_out.shape[1:]  # 除 batch 外的所有维度
            #     weight_mask = weight_mask.view(weight_mask.shape[0], *[1] * len(extra_dims))

            #     lora_output += expert_out * weight_mask
            lora_output = self.experts[self.active_adapter][0](x)

            output = base_output + lora_output
            return output

    def merge(self):
        return
    
    def unmerge(self):
        return
    
    def set_expert_weights(self, expert_weights, adapter_name=None):
        if adapter_name is None:
            adapter_name = self.active_adapter
        self.expert_weights[adapter_name] = expert_weights


class LoraExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, lora_alpha: float, lora_dropout: float, init_lora_weights: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = lora_alpha / r
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        
        if init_lora_weights:
            self.reset_lora_parameters()

    def forward(self, x):
        return self.lora_B(self.lora_dropout_layer(self.lora_A(x))) * self.scaling

    def reset_lora_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

