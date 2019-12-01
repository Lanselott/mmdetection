#!/usr/bin/env bash

set -x

PARTITION=gpu_24h
JOB_NAME=random_assign
CONFIG=./configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu_random_assign.py
WORK_DIR=./work/dirs/fcos_random_assign
GPUS=${5:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-"--validate"}
# CHECKPOINT_FILE=./work/dirs/fcos_random_assign/latest.pth

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work_dir=${WORK_DIR} --resume_from=${CHECKPOINT_FILE} --launcher="slurm" 
    #${PY_ARGS}
