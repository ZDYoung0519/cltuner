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
class SparseMoeLoraConfig(LoraConfig):
    expert_num: int = field(default=4)
    aux_loss_coef: int = field(default=1e-3)

    def __post_init__(self):
        self.peft_type = PeftType.MOELORA


class SparseMoeLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        # self.forward = self.model.forward
        self.peft_config = config
        self.moe_layers = []
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        total_aux_loss = sum(
            layer.get_aux_loss() or 0 
            for layer in self.moe_layers
        )
        if hasattr(outputs, 'loss') and total_aux_loss != 0:
            outputs.loss = outputs.loss + total_aux_loss
        return outputs

    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config
    
    def add_adapter(self, adapter_name, config=None):
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
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)
                self.moe_layers.append(new_module)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )
    
    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "expert_num": lora_config.expert_num,
            "aux_loss_coef ": lora_config.aux_loss_coef 
        }

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
            new_module = SparseMoeLoraLinear(
                adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module


class SparseMoeLoraLayer:
    def __init__(self, in_features: int, out_features: int, expert_num: int, **kwargs):
        self.experts = nn.ModuleDict({})
        self.router = nn.ModuleDict({})

        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.expert_num = expert_num
        self.kwargs = kwargs
        
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.experts.update(
            nn.ModuleDict({
                adapter_name: LoraExpert(self.in_features, self.out_features, r, lora_alpha, lora_dropout, init_lora_weights)})
            )
        self.router.update(
            nn.ModuleDict({adapter_name: Router(self.in_features, self.expert_num)})
        )
        self.to(self.weight.device)


class SparseMoeLoraLinear(nn.Linear, SparseMoeLoraLayer):
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
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.aux_loss_coef  = kwargs.pop("aux_loss_coef", 0)
        expert_num = kwargs.pop("expert_num", 8)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SparseMoeLoraLayer.__init__(self, in_features=in_features, out_features=out_features, expert_num=expert_num)
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
        previous_dtype = x.dtype
        if self.active_adapter not in self.experts.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:   # No adapter
            if self.merged: # merge the adapter to linear
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif not self.merged:
            base_output = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            batch_size, seq_len, _ = x.shape
            x = x.to(self.experts[self.active_adapter].lora_A.weight.dtype)
            expert_weights, expert_indices, router_probs = self.router[self.active_adapter](x)
            
            lora_output = torch.zeros_like(base_output)

            x_flat = x.view(-1, x.size(-1))
            expert_weights_flat = expert_weights.view(-1, self.num_experts_per_tok)
            expert_indices_flat = expert_indices.view(-1, self.num_experts_per_tok)

            for expert_idx in range(self.num_experts):

                mask = (expert_indices_flat == expert_idx).any(dim=-1)  # [B*S]
                if not mask.any():
                    continue
                
                expert_input = x_flat[mask]  # [num_selected, hidden]
                expert_out = self.experts[expert_idx](expert_input)  # [num_selected, out_features]
                
                weight_mask = (expert_indices_flat == expert_idx)  # [num_selected, K]
                weights = expert_weights_flat[mask] * weight_mask[mask].float()
                weights = weights.sum(dim=-1, keepdim=True)  # [num_selected, 1]
                lora_output.view(-1, lora_output.size(-1))[mask] += expert_out * weights
            
            output = base_output + lora_output.view(base_output.shape)
            if self.training and self.aux_loss_coef > 0:
                self._aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)
            else:
                self._aux_loss = 0
            self._router_probs = router_probs
            return output
    
    def _compute_load_balancing_loss(
        self, 
        router_probs: torch.Tensor, 
        expert_indices: torch.Tensor
    ) -> torch.Tensor:

        expert_mask = F.one_hot(
            expert_indices, 
            num_classes=self.num_experts
        ).sum(dim=-2)  # [B, S, K] -> sum over K -> [B, S, num_experts]
        avg_router_prob = router_probs.mean(dim=[0, 1])  # [num_experts]
        freq = expert_mask.float().mean(dim=[0, 1])  # [num_experts]
        aux_loss = self.num_experts * (freq * avg_router_prob).sum()
        return self.aux_loss_coef * aux_loss

    def merge(self):
        return
    
    def unmerge(self):
        return


class Router(nn.Linear):
    def __init__(self, in_features: int, expert_num: int, num_experts_per_tok: int=2):
        super().__init__(in_features, expert_num, bias=False)
    
    def forward(self, x):
        router_logits = torch.matmul(x, self.weight)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            router_probs, 
            self.num_experts_per_tok, 
            dim=-1
        )
        return expert_weights, expert_indices, router_probs


class LoraExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, lora_alpha: float, lora_dropout: float, init_lora_weights: bool):
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