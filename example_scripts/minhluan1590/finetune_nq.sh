DATA_DIR=./atlas_data
SIZE=xl
N_GPUS=8
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/nq_data/train.64-shot.jsonl"
EVAL_FILES="${DATA_DIR}/nq_data/dev.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=my-nq-64-shot-example
TRAIN_STEPS=30

# Use     --index_mode faiss --faiss_index_type ivfpq --faiss_code_size 16  if you don't want to use flat index
NCCL_DEBUG=INFO torchrun --nnodes=1 --nproc_per_node=${N_GPUS} --master_addr=localhost \
    finetune_qa.py \
    --shuffle \
    --train_retriever \
    --gold_score_mode pdist \
    --use_gradient_checkpoint_reader --use_gradient_checkpoint_retriever \
    --precision fp32 \
    --shard_optim --shard_grads \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --refresh_index -1 \
    --query_side_retriever_training \
    --target_maxlength 16 \
    --reader_model_type google/t5-${SIZE}-lm-adapt \
    --dropout 0.1 --weight_decay 0.01 --lr 4e-5 --lr_retriever 4e-5 --scheduler linear \
    --text_maxlength 512 \
    --model_path "${DATA_DIR}/models/atlas/${SIZE}" \
    --train_data "${DATA_DIR}/nq_data/train.64-shot.jsonl" \
    --eval_data "${DATA_DIR}/nq_data/dev.jsonl" \
    --per_gpu_batch_size 1 \
    --n_context 40 \
    --retriever_n_context 40 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq ${TRAIN_STEPS} \
    --log_freq 4 \
    --total_steps ${TRAIN_STEPS} \
    --warmup_steps 5 \
    --save_freq ${TRAIN_STEPS} \
    --main_port $port \
    --write_results \
    --task qa \
    --index_mode faiss --faiss_index_type ivfpq --faiss_code_size 16 \
    --passages "${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl" "${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl" \
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index