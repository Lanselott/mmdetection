#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=fcos_r50_caffe_fpn_gn_1x_4gpu_gradient_assign_nvidia_test.py
GPUS=4
WORK_DIR=/results

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch $WORK_DIR
