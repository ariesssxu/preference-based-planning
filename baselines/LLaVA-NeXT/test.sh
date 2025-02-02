#!/bin/bash
ROOT_DIR="."

if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="./ckpt/LLaVA-NeXT-Video-7B-DPO"
CONV_MODE="vicuna_v1"
FRAMES="32"
POOL_STRIDE="2"
OVERWRITE="True"
VIDEO_PATH="../../dataset/sample_task_1"


if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
# python3 test.py \
#     --model-path $CKPT \
#     --video_path ${VIDEO_PATH} \
#     --output_dir ./work_dirs/video_demo/$SAVE_DIR \
#     --output_name pred \
#     --chunk-idx $(($IDX - 1)) \
#     --overwrite ${OVERWRITE} \
#     --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
#     --for_get_frames_num $FRAMES \
#     --conv-mode $CONV_MODE 

# python3 test_few_shot.py \
#     --model-path $CKPT \
#     --video_path ${VIDEO_PATH} \
#     --output_dir ./work_dirs/video_demo/$SAVE_DIR \
#     --output_name pred \
#     --chunk-idx $(($IDX - 1)) \
#     --overwrite ${OVERWRITE} \
#     --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
#     --for_get_frames_num $FRAMES \
#     --conv-mode $CONV_MODE 

python3 test_few_shot_action.py \
    --model-path $CKPT \
    --video_path ${VIDEO_PATH} \
    --output_dir ./work_dirs/video_demo/$SAVE_DIR \
    --output_name pred \
    --chunk-idx $(($IDX - 1)) \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode $CONV_MODE 


