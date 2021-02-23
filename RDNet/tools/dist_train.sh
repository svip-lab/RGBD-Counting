#!/usr/bin/env bash

CONFIG=configs/retinanet_r101_fpn_1x_RGBD_700.py
GPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
