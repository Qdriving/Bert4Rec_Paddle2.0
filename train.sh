#!/bin/bash

set -xe

while true ; do
  case "$1" in
    -local) is_local="$2" ; shift 2 ;;
    *)
       if [[ ${#1} > 0 ]]; then
          echo "not supported arugments ${1}" ; exit 1 ;
       else
           break
       fi
       ;;
  esac
done

case "$is_local" in
    n) is_distributed="--is_distributed true" ;;
    y) is_distributed="--is_distributed false" ;;
    *) echo "not support argument -local: ${is_local}" ; exit 1 ;;
esac

#  pretrain config----
SAVE_STEPS=100
BATCH_SIZE=256
WARMUP_STEPS=100
NUM_TRAIN_STEPS=400000
LR_RATE=1e-3
WEIGHT_DECAY=0.01
MAX_LEN=200
TRAIN_DATA_DIR=bert_train/data/ml-1m-test.txt     #/home/aistudio/data/data94080/ml-20m-train.txt  #
VALIDATION_DATA_DIR=bert_train/data/ml-1m-test.txt     #/home/aistudio/data/data94080/ml-20m-test.txt  #
CONFIG_PATH=bert_train/bert_config_ml-1m_256.json  #bert_train/bert_config_ml-20m_256.json  #
VOCAB_PATH=data/demo_config/vocab.txt
INIT_DIR=./output/bert_best_1m.pdparams

# Change your train arguments:
python -u ./trainrec.py ${is_distributed}\
        --data_name ml-1m \
        --epoch 400 \
        --use_cuda true \
        --weight_sharing true\
        --batch_size ${BATCH_SIZE} \
        --warmup_steps ${WARMUP_STEPS} \
        --num_train_steps ${NUM_TRAIN_STEPS} \
        --data_dir ${TRAIN_DATA_DIR} \
        --validation_set_dir ${VALIDATION_DATA_DIR} \
        --bert_config_path ${CONFIG_PATH} \
        --vocab_path ${VOCAB_PATH} \
        --generate_neg_sample true\
        --checkpoints ./output \
        --save_steps ${SAVE_STEPS} \
        --learning_rate ${LR_RATE} \
        --weight_decay ${WEIGHT_DECAY:-0} \
        --max_seq_len ${MAX_LEN} \
        --skip_steps 20 \
        --validation_steps 1000 \
        --num_iteration_per_drop_scope 10 \
        --use_fp16 false \
        --loss_scaling 8.0 
      #  --init_checkpoint ${INIT_DIR} \
      #  --last_steps 390000 
       
