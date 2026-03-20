export BNB_CUDA_VERSION=124
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

CONFIG=projects/tadyra/scripts/ucit_llava_7b_v15pi_tadyra.py
CONFIG_ROUTING=projects/tadyra/scripts/ucit_llava_7b_v15pi_tadyra_train_routing.py

WORKDIR=work_dirs/tadyra/ucit_llava_7b_v15pi_tadyra
TASKID=0

bash projects/tadyra/scripts/train.sh 0 $CONFIG $CONFIG_ROUTING $WORKDIR
bash projects/tadyra/scripts/train.sh 1 $CONFIG $CONFIG_ROUTING $WORKDIR
bash projects/tadyra/scripts/train.sh 2 $CONFIG $CONFIG_ROUTING $WORKDIR
bash projects/tadyra/scripts/train.sh 3 $CONFIG $CONFIG_ROUTING $WORKDIR
bash projects/tadyra/scripts/train.sh 4 $CONFIG $CONFIG_ROUTING $WORKDIR
bash projects/tadyra/scripts/train.sh 5 $CONFIG $CONFIG_ROUTING $WORKDIR

