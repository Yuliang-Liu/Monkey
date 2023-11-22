
EVAL_PTH=$1
SAVE_NAME=$2




python -m torch.distributed.launch --use-env     --nproc_per_node ${NPROC_PER_NODE:-1}     --nnodes ${WORLD_SIZE:-1}     --node_rank ${RANK:-0} --master_addr ${MASTER_ADDR:-127.0.0.1}     --master_port ${MASTER_PORT:-12345}     eval/evaluate_vqa.py     --checkpoint $EVAL_PTH         --batch-size 4        --num-workers 2 --save_name $SAVE_NAME



