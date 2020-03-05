#!/usr/bin/env bash

set -x

PARTITION=gpu_24h
JOB_NAME=inception_tricksv5
CONFIG=./configs/fcos/ddb_v3_no_improvement_r50_caffe_fpn_gn_1x_single_test.py
WORK_DIR=./work/dirs/ddb_v3_no_improvement_single/
GPUS=${5:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-"--validate"}
# NOTE: first stage train 12 epoches
# CHECKPOINT_FILE=./work/dirs/ddb_v3_no_improvement_single/latest.pth

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
