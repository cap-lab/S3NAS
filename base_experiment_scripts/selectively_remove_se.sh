export PYTHONPATH=$PYTHONPATH:`pwd`

# extract se values
export DATA_DIR=WRITE_IMGS_DIR

export IMAGENET_SETTINGS="--log_excitation_names_containing=all --data_path=${DATA_DIR}"

export SAVE_FOLDER=WRITE_TMP_FOLDER
export IMG_SIZE=224

export TRAIN_DIR=WRITE_CKPT_DIR
export MODEL_NAME=WRITE_MODEL_NAME
export MODEL_JSON=models/${MODEL_NAME}.json
python etc_utils/extract_se_excitations.py --csv_save_dir=$SAVE_FOLDER --model_json_path=$MODEL_JSON --ckpt_dir=$TRAIN_DIR --input_image_size=$IMG_SIZE $IMAGENET_SETTINGS

# draw graph and extract se ranks
python etc_utils/excitation_csvs_to_rank_json.py --csv_folder=$SAVE_FOLDER

# remove with se ranks
export BLOCK_NUMS=2
export SUFFIX=_rm_stdavg_${BLOCK_NUMS}
export DEFAULT_SETTINGS="--mode=select_by_rank --se_ratio=0 --rank_json=${SAVE_FOLDER}/ranks.json"
python etc_utils/set_se_actfn_to_json.py $MODEL_JSON --suffix=$SUFFIX --set_blocks_num=$BLOCK_NUMS $DEFAULT_SETTINGS
