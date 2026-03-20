conda activate my-cltuner-env
export BNB_CUDA_VERSION=124
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

bash projects/dyra/scripts/train.sh 0
bash projects/dyra/scripts/train.sh 1
bash projects/dyra/scripts/train.sh 2
bash projects/dyra/scripts/train.sh 3
bash projects/dyra/scripts/train.sh 4
bash projects/dyra/scripts/train.sh 5
