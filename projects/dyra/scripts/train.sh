TASKID=$1
CONFIG=projects/dyra/scripts/ucit_llava_7b_v15pi.py
WORKDIR=work_dirs/dyra/ucit_llava_7b_v15pi_dyra


if [ "$TASKID" -gt 0 ]; then
    LASTCKPT=$(cat $WORKDIR/task$((TASKID-1))/last_checkpoint)
else
    LASTCKPT=$(python -c "import mmengine; config = mmengine.Config.fromfile('$CONFIG'); print(config.pretrained_pth)")
fi

echo "######################################################"
echo "############## Train Task $TASKID  ###################"
echo "######################################################"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

echo "Using Devices: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NGPUS"
echo "Config file: $CONFIG"
echo "Work dir: $WORKDIR"
echo "Checkpoint: $LASTCKPT"

# train
# torchrun --nproc_per_node=$NGPUS cltuner/tools/train.py \
#     $CONFIG  \
#     --cur-task $TASKID \
#     --cfg-options model.pretrained_pth=$LASTCKPT \
#         model.cur_task=$TASKID \
#         model.llm_lora.cur_task=$TASKID \
#         model.llm_lora.lora_init_file=work_dirs/tadyra/ucit_llava_7b_v15pi_tadyra/task0/dyra_init.pt \
#     --work-dir $WORKDIR/task$TASKID \
#     --launcher pytorch \
#     --deepspeed deepspeed_zero2

# get the trained checkpoint
CKPT=$(cat $WORKDIR/task$TASKID/last_checkpoint)

# generate answers on all encountered tasks
for EVALTASKID in $(seq 0 $TASKID); do
    OUTPUTDIR=$WORKDIR/eval/task$TASKID/task$EVALTASKID
    for IDX in $(seq 0 $((NGPUS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python cltuner/tools/generate.py \
            $CONFIG \
            --num-chunks $NGPUS  \
            --chunk-idx $IDX \
            --output-dir $OUTPUTDIR  \
            --eval-task $EVALTASKID \
            --cfg-options model.pretrained_pth=$CKPT \
                model.llm_lora.lora_init_file=work_dirs/tadyra/ucit_llava_7b_v15pi_tadyra/task0/dyra_init.pt &
    done
    wait
    cat "${OUTPUTDIR}"/chunk*.jsonl > "${OUTPUTDIR}/merge.jsonl"
done

python cltuner/tools/eval/eval_entry_cl.py $CONFIG --cur-task $TASKID --work-dir $WORKDIR
