TASKID=$1
CONFIG=$2
CONFIG_ROUTING=$3
WORKDIR=$4


IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

echo "Using Devices: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NGPUS"
echo "Config file: $CONFIG"
echo "Work dir: $WORKDIR"

echo "######################################################"
echo "############## Train Task $TASKID  ###################"
echo "######################################################"


# train dynamic routing
torchrun --nproc_per_node=$NGPUS projects/tadyra/tools/train_dynamic_routing.py \
    $CONFIG_ROUTING  \
    --cur-task $TASKID \
    --work-dir $WORKDIR \
    --launcher pytorch
CKPT_ROUTING=$WORKDIR/task$TASKID/dynamic_routing_best.pth

# dynamic ranking
torchrun --nproc_per_node=$NGPUS projects/tadyra/tools/train_dynamic_ranking.py \
    $CONFIG \
    --cur-task $TASKID \
    --work-dir $WORKDIR \
    --launcher pytorch \
    --deepspeed deepspeed_zero2
DYRA_INIT=$WORKDIR/task$TASKID/dyra_init.pt

if [ "$TASKID" -gt 0 ]; then
    LASTCKPT=$(cat $WORKDIR/task$((TASKID-1))/last_checkpoint)
else
    LASTCKPT=$(python -c "import mmengine; config = mmengine.Config.fromfile('$CONFIG'); print(config.pretrained_pth)")
fi


echo "Checkpoint: $LASTCKPT"

# train
echo torchrun --nproc_per_node=$NGPUS cltuner/tools/train.py \
    $CONFIG  \
    --cur-task $TASKID \
    --work-dir $WORKDIR/task$TASKID \
    --cfg-options model.pretrained_pth=$LASTCKPT \
        model.llm_lora.lora_init_file=$DYRA_INIT \
        model.llm_lora.cur_task=$TASKID \
        model.cur_task=$TASKID \
    --launcher pytorch \
    --deepspeed deepspeed_zero2

# # get the trained path
# CKPT=$(cat $WORKDIR/task$TASKID/last_checkpoint)

# # generate answers on all encountered tasks
# for EVALTASKID in $(seq 0 $TASKID); do
#     OUTPUTDIR=$WORKDIR/eval/task$TASKID/task$EVALTASKID
#     for IDX in $(seq 0 $((NGPUS-1))); do
#         CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python cltuner/tools/generate.py \
#             $CONFIG \
#             --num-chunks $NGPUS  \
#             --chunk-idx $IDX \
#             --output-dir $OUTPUTDIR  \
#             --eval-task $EVALTASKID \
#             --cfg-options model.pretrained_pth=$CKPT &
#     done
#     wait
# done

# python cltuner/tools/eval/eval_entry_cl.py $CONFIG --cur-task $TASKID --work-dir $WORKDIR
