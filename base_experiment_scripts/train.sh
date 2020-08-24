# Directory settings
export STORAGE_BUCKET=gs://WRITE_BUCKET_NAME
export DATA_DIR=${STORAGE_BUCKET}/imagenet

export MODEL_NAME=WRITE_THE_SUPERGRAPH_NAME_HERE
export MODEL_JSON=models/${MODEL_NAME}.json

export TEST_NAME=${MODEL_NAME}
export TEST_SAVE_FOLDER=${STORAGE_BUCKET}
export TRAIN_DIR=${TEST_SAVE_FOLDER}/${TEST_NAME}_train

# tpu
export TPU_SETTINGS=--tpu=WRITE_YOUR_TPU_NAME_HERE

# Common settings
export TRAIN_BATCH_SIZE=1024
export EVAL_BATCH_SIZE=1000
export DROPOUT=WRITE_DROPOUT_RATE

export INPUT_IMAGE_SIZE=WRITE_IMAGE_SIZE

export PYTHONPATH=$PYTHONPATH:$(pwd)

# train result
export TRAIN_SETTINGS="--model_json_path=${MODEL_JSON} --input_image_size=$INPUT_IMAGE_SIZE --model_dir=$TRAIN_DIR --train_batch_size=$TRAIN_BATCH_SIZE --eval_batch_size=$EVAL_BATCH_SIZE --dropout_rate=$DROPOUT --data_dir=$DATA_DIR"
export TRAIN_LOG_FILE=${TEST_NAME}_train.log
python run/main.py --train_epochs=120 --epochs_per_eval=40 $TPU_SETTINGS $TRAIN_SETTINGS 2>>$TRAIN_LOG_FILE
python run/main.py --train_epochs=350 --epochs_per_eval=5 $TPU_SETTINGS $TRAIN_SETTINGS 2>>$TRAIN_LOG_FILE
gsutil cp ${TRAIN_LOG_FILE} ${TRAIN_DIR}/${TRAIN_LOG_FILE}
