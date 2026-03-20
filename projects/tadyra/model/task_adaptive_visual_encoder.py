
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmengine.device import get_device
from mmengine import print_log
from xtuner.registry import BUILDER
from xtuner.model.utils import LoadWoInit, traverse_dict
from mmengine.model import BaseModel

from .modules import TaskAdaptiveNullSpaceAttention



class DynamicRouting(nn.Linear):
    def __init__(self, hidden_dim, num_experts, bias=True, temperature=0.1):
        super().__init__(hidden_dim, num_experts, bias)
        self.temperature = temperature
    
    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        return x / self.temperature


class TaskAdaptiveVisualEncoder(BaseModel):
    def __init__(
        self,
        visual_encoder,
        text_encoder,
        text_tokenizer,
        visual_select_layers=None,
        cur_task=0,
        num_experts=10,
        task_inference_temp=0.1,
    ):
        super().__init__()

        with LoadWoInit():
            self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)
            self.text_encoder = self._build_from_cfg_or_module(text_encoder)
        
        if isinstance(text_tokenizer, dict):
            self.text_tokenizer = self._build_from_cfg_or_module(text_tokenizer)
        else:
            self.text_tokenizer = text_tokenizer
        
        self.task_adaptive_attention = TaskAdaptiveNullSpaceAttention(
            self.visual_encoder.config.hidden_size,
            self.text_encoder.config.hidden_size,
            visual_select_layers
        )

        self.dynamic_routring = DynamicRouting(
            hidden_dim=self.visual_encoder.config.hidden_size + self.text_encoder.config.hidden_size,
            num_experts=num_experts,
            temperature=task_inference_temp
        )

        self.cur_task = cur_task
        self.forward_dynamic_routring_only = False
        self.return_loss = False

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError
    
    def forward(self, data, data_samples, mode='loss'):
        if "pixel_values" not in data:
            return data

        images = data['pixel_values']
        texts = data['text']
        dtype = images.dtype
        device = get_device()

        visual_outputs = self.visual_encoder(
            images.to(dtype).to(device), 
            output_hidden_states=True
        )
        clip_text_inputs = self.text_tokenizer(
            texts,
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        clip_text_inputs.to(device)
        text_outputs = self.text_encoder(**clip_text_inputs)

        # concat visual and test cls token
        embeddings = torch.cat(
            [
                visual_outputs['last_hidden_state'][:, 0,:], 
                text_outputs['last_hidden_state'][:, 0, :]
            ],
            dim=-1
        )
        expert_weights, loss = self._get_expert_weights(embeddings.detach())
        if mode=="loss":
            return {"loss": loss}
        
        if self.forward_dynamic_routring_only:
            output = {
                "embeddings": embeddings,
                "expert_weights": expert_weights,
                "l_task": loss
            }
        else:
            attention, fused_features = self.task_adaptive_attention(visual_outputs, text_outputs)
            output = {
                "visual_outputs": visual_outputs,
                "text_outputs": text_outputs,
                "taa_attention": attention,
                "taa_vision_features": fused_features,
                "l_taa": self._get_taa_loss(attention),
                "embeddings": embeddings,
                "expert_weights": expert_weights,
                "l_task": loss
            }
        for k, v in output.items():
            data[k] = v
        return data

    def _get_expert_weights(self, embeddings, mask_logits=True):
        cur_task = self.cur_task
        
        logits = self.dynamic_routring(embeddings)   # B, T
        if mask_logits:
            logits[:, (cur_task+1):] = float('-inf')

        task_probs = nn.Softmax(dim=-1)(logits)

        if self.return_loss:
            targets = torch.LongTensor([cur_task] * logits.shape[0]).to(logits.device)
            l_task = nn.CrossEntropyLoss()(logits, targets)
        else:
            l_task = 0
        return task_probs.detach(), l_task

    def _get_taa_loss(self, attention):
        attention = attention.squeeze(1)    # B, K
        l_taa = (attention * torch.log(attention)).sum(dim=-1).sum()
        return l_taa
    

    