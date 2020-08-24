# Directory settings
export STORAGE_BUCKET=gs://WRITE_BUCKET_NAME
export DATA_DIR=${STORAGE_BUCKET}/imagenet

export MODEL_NAME=WRITE_THE_SUPERGRAPH_NAME_HERE
export MODEL_JSON=models/supergraphs/${MODEL_NAME}.json

export TEST_NAME=${MODEL_NAME}
export TEST_SAVE_FOLDER=${STORAGE_BUCKET}

# tpu
export TPU_SETTINGS=--tpu=WRITE_YOUR_TPU_NAME_HERE

# Common settings
export SEARCH_BATCH_SIZE=1024
export TRAIN_BATCH_SIZE=1024
export EVAL_BATCH_SIZE=1000
export DROPOUT=0.2

export INPUT_IMAGE_SIZE=224
export TARGET_LATENCY=WRITE_TARGET_LATENCY

export PYTHONPATH=$PYTHONPATH:$(pwd)

# search setting
export SUPERGRAPH_TRAIN_EPOCHS=8
export TOTAL_SERACH_EPOCHS=10
export DUMMY_EPOCH_TO_AVOID_TENSORBOARD_ERROR=$(($SUPERGRAPH_TRAIN_EPOCHS + 2))

export SEARCH_DIR=${TEST_SAVE_FOLDER}/${TEST_NAME}_search
export TRAIN_DIR=${TEST_SAVE_FOLDER}/${TEST_NAME}_train
export BACKUP_DIR=${SEARCH_DIR}_${SUPERGRAPH_TRAIN_EPOCHS}epoch

export COMMON_SEARCH_SETTINGS="--constraint_lut_folder=WRITE_LATENCY_FOLDER --drop_connect_rate=0 --use_nas_modelmaker=True --epochs_per_eval=10 --train_batch_size=$SEARCH_BATCH_SIZE --eval_batch_size=$EVAL_BATCH_SIZE --model_json_path=$MODEL_JSON --model_dir=${SEARCH_DIR} --target_latency=$TARGET_LATENCY --dropout_rate=$DROPOUT --data_dir=$DATA_DIR --input_image_size=$INPUT_IMAGE_SIZE"
export SEARCH_SETTINGS="--log_searchableblock_tensor=never --train_epochs=$SUPERGRAPH_TRAIN_EPOCHS --supergraph_train_epochs=$DUMMY_EPOCH_TO_AVOID_TENSORBOARD_ERROR"

export SEARCH_LOG_FILE=${TEST_NAME}_search.log

# supergraph train
python run/main.py --clip_gradients=10.0 $TPU_SETTINGS $SEARCH_SETTINGS $COMMON_SEARCH_SETTINGS 2>> $SEARCH_LOG_FILE
gsutil cp ${SEARCH_LOG_FILE} ${SEARCH_DIR}/${SEARCH_LOG_FILE}

# backup
gsutil -m cp -r ${SEARCH_DIR} $BACKUP_DIR

# complete search
export SEARCH_SETTINGS="--train_epochs=$TOTAL_SERACH_EPOCHS --supergraph_train_epochs=$SUPERGRAPH_TRAIN_EPOCHS"
python run/main.py $TPU_SETTINGS $SEARCH_SETTINGS $COMMON_SEARCH_SETTINGS 2>>${SEARCH_LOG_FILE}
gsutil cp ${SEARCH_LOG_FILE} ${SEARCH_DIR}/${SEARCH_LOG_FILE}

# parse search result
export PARSE_LOG_FILE=${TEST_NAME}_parse.log
python graph/print_parsed_args.py --parse_search_dir=${SEARCH_DIR} --search_model_json_path=$MODEL_JSON 2>> $PARSE_LOG_FILE
python etc_utils/print_latency_from_tb.py --tensorboard_dir=${SEARCH_DIR} 2>> $PARSE_LOG_FILE
gsutil cp $PARSE_LOG_FILE ${SEARCH_DIR}/${PARSE_LOG_FILE}

# train result
export TRAIN_SETTINGS="--model_json_path=${SEARCH_DIR}/parsed_model.json --input_image_size=$INPUT_IMAGE_SIZE --model_dir=$TRAIN_DIR --train_batch_size=$TRAIN_BATCH_SIZE --eval_batch_size=$EVAL_BATCH_SIZE --dropout_rate=$DROPOUT --data_dir=$DATA_DIR"
export TRAIN_LOG_FILE=${TEST_NAME}_train.log
python run/main.py --train_epochs=120 --epochs_per_eval=40 $TPU_SETTINGS $TRAIN_SETTINGS 2>>$TRAIN_LOG_FILE
python run/main.py --train_epochs=350 --epochs_per_eval=5 $TPU_SETTINGS $TRAIN_SETTINGS 2>>$TRAIN_LOG_FILE
gsutil cp ${TRAIN_LOG_FILE} ${TRAIN_DIR}/${TRAIN_LOG_FILE}
