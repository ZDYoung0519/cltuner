#######################################################################
#                             Data Path                               #
#######################################################################

# UCIT
data_root_ucit = "/storage/clbenchmarks/UCIT/UCIT/"
data_root_ucit_offline = "/storage/clbenchmarks/UCIT/UCIT_llamma2_tokenized/"
image_folder_ucit = "/storage/public_datasets/"

# COIN


# ...


#######################################################################
#                             Model Path                              #
#######################################################################

# CLIP
clip_vit_large_p14_336 = "/storage/huggingface/openai/clip-vit-large-patch14-336"

# LLaVA v1.5 model with pre-trained and instruction-fintuned weights
llm_llava_v15pi = "/storage/huggingface/llava-v1.5-7b-xtuner"                                  # llm weights
projector_llava_v15pi = "/storage/huggingface/llava-v1.5-7b-xtuner/mm_projector_xtuner.pt"     # mlp weights

# LLaVA v1.5 model with pre-trained weights
llm_llava_v15p = "/storage/huggingface/lmsys/vicuna-7b-v1.5"
projector_llava_v15p = "/storage/huggingface/llava-v1.5-pretrained/mm_projector_xtuner.pt"

# LLaVA v1.5 model without pre-trained and instruction-tuned
llm_llava_v15 = "/storage/huggingface/lmsys/vicuna-7b-v1.5"
projector_llava_v15 = None


