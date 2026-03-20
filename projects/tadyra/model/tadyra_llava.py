import torch

from collections import OrderedDict

from xtuner.registry import BUILDER
from xtuner.model.utils import LoadWoInit, make_inputs_require_grad, prepare_inputs_labels_for_multimodal
from cltuner.model.llava import LLaVAModel, XTunerLLaVAModel, get_peft_model_state_dict

from .modules import NSProjectorModel, ProjectorConfig
from .task_adaptive_visual_encoder import TaskAdaptiveVisualEncoder


class TadyraLLaVAModel(LLaVAModel):
    def __init__(
        self,
        llm,
        visual_encoder,
        freeze_llm=False,
        freeze_visual_encoder=False,
        visual_select_layer=-2,
        pretrained_pth=None,
        projector_depth=2,
        llm_lora=None,
        visual_encoder_lora=None,
        use_activation_checkpointing=True,
        max_position_embeddings=None,
        # add args below
        cur_task=0,
        num_experts=6,
        text_encoder=None,
        text_tokenizer=None,
        taa_select_layers=None,
        taa_add_residual=True,
        taa_loss_ceof=1e-4,
        taa_pretrained_pth="",
        freeze_projector_bias=False,
        train_cur_lora_only=True,
        disable_taa = True,
        **kwargs
    ):
        super().__init__(
            llm,
            visual_encoder, 
            freeze_llm, 
            freeze_visual_encoder, 
            visual_select_layer, 
            None,
            projector_depth,
            llm_lora,
            visual_encoder_lora,
            use_activation_checkpointing,
            max_position_embeddings,
            **kwargs
        )

        with LoadWoInit():
            self.text_encoder = self._build_from_cfg_or_module(text_encoder)
            self.text_tokenizer = self._build_from_cfg_or_module(text_tokenizer)
        
        # freeze text_encoder 
        self.text_encoder.requires_grad_(False)
        if use_activation_checkpointing:
            if hasattr(self.text_encoder, "enable_input_require_grads"):
                self.text_encoder.enable_input_require_grads()
            else:
                self.text_encoder.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )
            self.text_encoder.gradient_checkpointing_enable()
        
        self.taa_visual_encoder = TaskAdaptiveVisualEncoder(
            self.visual_encoder,
            self.text_encoder,
            self.text_tokenizer,
            visual_select_layers=taa_select_layers,
            cur_task=cur_task,
            num_experts=num_experts
        )

        # freeze routing as it's already trained
        for n, p in self.taa_visual_encoder.named_parameters():
            if "dynamic_routring." in n:
                p.requires_grad_(False)

        projector_config = ProjectorConfig(
            visual_hidden_size=self.visual_encoder.config.hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=self.projector_depth,
            bias=True
        )
        self.taa_projector = NSProjectorModel(projector_config).to(self.visual_encoder.dtype)
        self.taa_add_residual = taa_add_residual
        self.taa_loss_ceof = taa_loss_ceof
        self.disable_taa = disable_taa
        self.cur_task = cur_task
        self.num_experts = num_experts

        # freeze projector bias
        if freeze_projector_bias:
            for n, p in self.taa_projector.named_parameters():
                if '.bias' in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
            for n, p in self.projector.named_parameters():
                if '.bias' in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)

        if taa_pretrained_pth:
            self.load_checkpoint(taa_pretrained_pth)

        self.load_checkpoint(pretrained_pth)

        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.shape)
        
        # for n, p in self.state_dict().items():
        #     print(n, p.shape)

    def _prepare(self, data):
        if not self.disable_taa:
            # aggregate vision features
            data = self.taa_visual_encoder(data, None, mode="")
            taa_vision_features = data['taa_vision_features']
            taa_vision_features = self.taa_projector(taa_vision_features)
            if self.taa_add_residual:
                original_selected_vision_features = data['visual_outputs'].hidden_states[self.visual_select_layer][:, 1:]
                original_selected_vision_features = self.projector(original_selected_vision_features)
                taa_vision_features = taa_vision_features + original_selected_vision_features
            data['pixel_values'] = taa_vision_features
            # aux loss for taa
            taa_attention = data.pop("taa_attention")
            self.l_taa = self._get_taa_loss(taa_attention) if self.taa_loss_ceof > 0 else 0
            expert_weights = data.pop('expert_weights')
        else:
            data = super()._prepare(data)
            self.l_taa = 0
            bs = data['pixel_values'].shape[0]
            expert_weights = torch.zeros(bs, self.num_experts)
            expert_weights[:, self.cur_task] = 1
            expert_weights.to(data['pixel_values']).device
            
        # set expert weight for llm lora
        if hasattr(self.llm, "set_expert_weights"):
            self.llm.set_expert_weights(expert_weights)
        return data

    def _get_taa_loss(self, attention):
        attention = attention.squeeze(1)    # B, K
        l_aux = (attention * torch.log(attention)).sum(dim=-1).sum()
        return l_aux

    def forward(self, data, data_samples=None, mode="loss"):
        if self.is_first_iter:
            self.to(data["input_ids"].device)
            self.is_first_iter = False

        if "pixel_values" in data:
            data = self._prepare(data)
            data = {
                "input_ids": data.get("input_ids", None),
                "position_ids": data.get("position_ids", None),
                "attention_mask": data.get("attention_mask", None),
                "past_key_values": data.get("past_key_values", None),
                "labels": data.get("labels", None),
                "pixel_values": data.get("pixel_values", None)
            }
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == "loss":
            return self.compute_loss(data, data_samples)
        elif mode == "predict":
            return self.predict(data, data_samples)
        elif mode == "tensor":
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError
    
    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {"loss": outputs.loss + self.l_taa}
        return loss_dict

    def state_dict(self, *args, **kwargs):
        """
        XTunerLLaVAModel only support `LoRA` finetune.
        We have to modify this to support differnt peft types
        """
        state_dict = super(XTunerLLaVAModel, self).state_dict()
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(self.visual_encoder, state_dict=state_dict)
            )
        elif not self.freeze_visual_encoder:
            to_return.update(
                {k: v for k, v in state_dict.items() if "visual_encoder." in k}
            )
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({k: v for k, v in state_dict.items() if "llm." in k})
        # Step 3. Projector
        to_return.update({k: v for k, v in state_dict.items() 
                          if "task_adaptive_attention." in k 
                          or "projector." in k
                          or "taa_projector." in k
                        })
        return to_return
    