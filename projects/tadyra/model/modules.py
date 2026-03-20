import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from xtuner.model.llava import ProjectorConfig


class NSLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)


class NS2Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
    
    def forward(self, x1, x2):
        """
        x1: (batch_size, n1, dim1)
        x2: (batch_size, n2, dim2)
        out: (batch_size, n1, n2)
        """
        x1 = F.linear(x1, self.weight.T, self.bias)    # b, n2, d1
        out = torch.bmm(x1, x2.transpose(-2, -1))      # b, n1, n2
        return out


class NSProjectorModel(PreTrainedModel):
    _auto_class = "AutoModel"
    config_class = ProjectorConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        modules = [
            NSLinear(
                config.visual_hidden_size, config.llm_hidden_size, bias=config.bias
            )
        ]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                NSLinear(
                    config.llm_hidden_size, config.llm_hidden_size, bias=config.bias
                )
            )
        self.model = nn.Sequential(*modules)

    def enable_input_require_grads(self):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, NSProjectorModel):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs


class TaskAdaptiveNullSpaceAttention(nn.Module):
    def __init__(self, visual_hidden_size, text_hidden_size, visual_select_layers):
        super().__init__()
        self.w1 = NS2Linear(visual_hidden_size, text_hidden_size, bias=False)
        self.w2 = NSLinear(visual_hidden_size, visual_hidden_size, bias=False)

        self.visual_select_layers = visual_select_layers
        self.linears = nn.ModuleList([])
        for _ in range(len(visual_select_layers)):
            self.linears.append(NSLinear(visual_hidden_size, visual_hidden_size, bias=False))

    def _collect_visual_features(self, visual_outputs, i):
        layer = self.visual_select_layers[i]
        if isinstance(layer, str) and '-' in layer:
            start, end = layer.split('-')
            start, end = int(start), int(end)
            feature = [visual_outputs.hidden_states[idx] for idx in range(start, end+1)]
            feature = torch.stack(feature, dim=0).mean(dim=0)   # avg pooling
        else:
            idx = int(layer)
            feature = visual_outputs.hidden_states[idx]
        feature = self.linears[i](feature)
        return feature
    
    def forward(self, visual_outputs, text_outputs):
        text_features = text_outputs['last_hidden_state']                      # B, L, d 
        selected_vision_features = [
            self._collect_visual_features(visual_outputs, i) for i in range(len(self.visual_select_layers))
        ]
        vision_features = torch.stack(selected_vision_features, dim=1).to(text_features.dtype)   # B, K, L_v ,d
        vision_features = vision_features[:, :, 1:, :]

        # attention-based fusion
        query = text_features[:, 0, :].unsqueeze(1)    # B, 1, d'
        key = vision_features[:, :, 0, :]              # B, K, d
        value = vision_features[:, :, :, :]            # B, K, L, d
        attention = self.w1(query, key)                # (B, 1, d') * (B, K, d) -> (B, 1, K)
        attention = nn.Softmax(dim=-1)(attention / vision_features.shape[-1]**0.5)    # (B, 1, K)

        _attention = attention.transpose(1, 2).unsqueeze(-1)         # (B, 1, K) -> (B, K, 1) -> (B, K, 1, 1)
        fused_features = (value * _attention).sum(dim=1)    # (B, K, L, d) * (B, K, 1, 1) -> (B, K, L, d) -> (B, L, d)
        fused_features = self.w2(fused_features)              # (B, L, d)
        return attention, fused_features

