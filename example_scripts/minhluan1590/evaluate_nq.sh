DATA_DIR=./atlas_data
SIZE=xl
N_GPUS=8
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/nq_data/train.64-shot.jsonl"
EVAL_FILES="${DATA_DIR}/nq_data/dev.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=my-nq-64-shot-example
TRAIN_STEPS=30

torchrun --nnodes=1 --nproc_per_node=${N_GPUS} --master_addr=localhost \
    evaluate.py \
    --name 'my-nq-64-shot-example-evaluation' \
    --generation_max_length 16 \
    --gold_score_mode "pdist" \
    --precision fp32 \
    --reader_model_type google/t5-${SIZE}-lm-adapt \
    --text_maxlength 512 \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}/checkpoint/step-30 \
    --eval_data "${DATA_DIR}/nq_data/dev.jsonl" "${DATA_DIR}/nq_data/test.jsonl" \
    --per_gpu_batch_size 1 \
    --n_context 40 --retriever_n_context 40 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode faiss --faiss_index_type ivfpq --faiss_code_size 16 \
    --task "qa" \
    --load_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index \
    --write_results