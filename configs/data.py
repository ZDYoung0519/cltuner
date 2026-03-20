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
llm_llava_v15_7b = "/storage/huggingface/llava-v1.5-7b-xtuner"                                  # llm weights
projector_llava_v15_7b = "/storage/huggingface/llava-v1.5-7b-xtuner/mm_projector_xtuner.pt"     # mlp weights

# LLaVA v1.5 model with pre-trained weights
llm_vicuna_v15_7b = "/storage/huggingface/lmsys/vicuna-7b-v1.5"
projector_vicuna_v15_7b_pretrain = "/storage/huggingface/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"

# projector_vicuna_v15_7b = "/storage/huggingface/llava-v1.5-pretrained/mm_projector_xtuner.pt"



