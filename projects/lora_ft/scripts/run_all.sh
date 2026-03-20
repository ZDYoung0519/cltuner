export BNB_CUDA_VERSION=124
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

CONFIG=projects/lora_ft/scripts/ucit_llava_7b_v15pi_lora.py
WORKDIR=work_dirs/lora_ft/ucit_llava_7b_v15pi_lora

bash projects/lora_ft/scripts/train.sh 0 $CONFIG $WORKDIR
bash projects/lora_ft/scripts/train.sh 1 $CONFIG $WORKDIR
bash projects/lora_ft/scripts/train.sh 2 $CONFIG $WORKDIR
bash projects/lora_ft/scripts/train.sh 3 $CONFIG $WORKDIR
bash projects/lora_ft/scripts/train.sh 4 $CONFIG $WORKDIR
bash projects/lora_ft/scripts/train.sh 5 $CONFIG $WORKDIR

