import torch
from torch import nn
from mmengine import print_log
from xtuner.model.utils import LoadWoInit, traverse_dict, prepare_inputs_labels_for_multimodal
from cltuner.model.llava import LLaVAModel

from peft.tuners.dyra import mark_only_cur_lora_as_trainable


class DyraLLaVAModel(LLaVAModel):
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
        text_encoder=None,
        text_tokenizer=None,
        taa_args={},
        router_args={},
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
        self.cur_task = cur_task

        with LoadWoInit():
            if text_encoder:
                self.text_encoder = self._build_from_cfg_or_module(text_encoder)
                for n, p in self.text_encoder.named_parameters():
                    p.requires_grad_(False)
            if text_tokenizer:
                self.text_tokenizer = self._build_from_cfg_or_module(text_tokenizer)
            
        # self._setup_taa(**taa_args)
        self._setup_router(**router_args)
        
        mark_only_cur_lora_as_trainable(self.llm, cur_task=cur_task)

        count = 0
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.shape)
                count += p.numel()
        print_log(f"Trainable Parameters: {count/1024/1024} M", "current")

        self.load_checkpoint(pretrained_pth)
    
    def _setup_taa(self, text_encoder, text_tokenizer):
        pass

    def _setup_router(
            self, 
            router_bias: bool = False, 
            router_temp: float = 1.0,
            use_trained_routing: bool = True, 
            trained_router_path: bool = False,
            router_loss_ceof = 1e-3,
        ):
        self.router = nn.Linear(
            in_features=self.visual_encoder.config.hidden_size + self.text_encoder.config.hidden_size,
            out_features=self.llm.peft_config['default'].num_experts,
            bias=router_bias
        )
        self.router_criterion = nn.CrossEntropyLoss()
        self.router_temp = router_temp
        self.router_loss_ceof = router_loss_ceof

        # if use_trained_routing, we load weights from pth and freeze them
        if use_trained_routing and trained_router_path:
            for _, p in self.router.named_parameters():
                p.requires_grad_(False)
            self.router.load_state_dict(
                torch.load(trained_router_path, map_location='cpu')
            )
            self.use_trained_routing = True
        self.use_trained_routing = False

    def _prepare(self, data):
        device = data["pixel_values"].device
        dtype = self.visual_encoder.dtype

        visual_outputs = self.visual_encoder(
            data["pixel_values"].to(dtype),
            output_hidden_states=True,
        )

        clip_text_inputs = self.text_tokenizer(
            data['text'],
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        clip_text_inputs.to(device)
        text_outputs = self.text_encoder(**clip_text_inputs)

        # prepare visual inputs
        pixel_values = self._get_pixel_values(visual_outputs, text_outputs)
        data["pixel_values"] = pixel_values

        # get expert weight for llm lora
        expert_weights, router_loss = self._get_expert_weights(visual_outputs, text_outputs)
        self.llm._set_expert_weights(expert_weights)
        self.router_loss = router_loss * self.router_loss_ceof if not self.use_trained_routing else 0
        return data

    def _get_pixel_values(self, visual_outputs, text_outputs=None):
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        )
        return pixel_values


    def _get_expert_weights(self, visual_outputs, text_outputs):
        visual_features = visual_outputs.last_hidden_state[:, 0, :]
        text_features = text_outputs.last_hidden_state[:, 0, :]
        features = torch.cat([visual_features, text_features], dim=-1).detach()
        logits = self.router(features) / self.router_temp
        logits[:, (self.cur_task+1):] = float('-inf')

        targets = torch.LongTensor([self.cur_task] * logits.shape[0]).to(logits.device)
        loss = self.router_criterion(logits, targets)            # B, T
        with torch.no_grad():
            expert_weights = nn.Softmax(dim=-1)(logits).mean(0).detach()
        return expert_weights, loss
    

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
        loss_dict = {"loss": outputs.loss + self.router_loss}
        return loss_dict