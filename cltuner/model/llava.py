from collections import OrderedDict
from xtuner.model.llava import LLaVAModel as XTunerLLaVAModel
from xtuner.model.utils import prepare_inputs_labels_for_multimodal, guess_load_checkpoint
from mmengine import print_log


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    bias = config.bias
    
    if bias == "none":
        to_return = {k: state_dict[k] for k in state_dict if "lora" in k}
    elif bias == "all":
        to_return = {
            k: state_dict[k] for k in state_dict if "lora" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {
        k: v
        for k, v in to_return.items()
        if (("lora_" in k and adapter_name in k) or ("bias" in k))
    }

    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(
                f"{module_name}.modules_to_save.{adapter_name}" in key
                for module_name in model.modules_to_save
            ):
                to_return[key] = value
    return to_return


class LLaVAModel(XTunerLLaVAModel):
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
        self.load_checkpoint(pretrained_pth)

    def _prepare(self, data):
        visual_outputs = self.visual_encoder(
            data["pixel_values"].to(self.visual_encoder.dtype),
            output_hidden_states=True,
        )
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        )
        data["pixel_values"] = pixel_values
        return data

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

    def load_checkpoint(self, checkpoint_pth):
        if checkpoint_pth is not None:
            state_dict = guess_load_checkpoint(checkpoint_pth)
            self.load_state_dict(state_dict, strict=False)
            print_log(f"Load checkpoint from {checkpoint_pth} successfully!", 'current')

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
        to_return.update({k: v for k, v in state_dict.items() if "projector." in k})
        return to_return
