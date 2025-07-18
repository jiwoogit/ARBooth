#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

GPU_ID=0
LOCAL_OUT=local_run
exp_name=brown_cat
cls_name=cat

mkdir -p $LOCAL_OUT
data_path="./inputs/${exp_name}"
val_prompt_path='prompts_validate.txt'
local_out_path=$LOCAL_OUT/${exp_name}

CUDA_VISIBLE_DEVICES=${GPU_ID} python \
booth_train_final.py \
--instance_prompt="<placeholder> ${cls_name}" \
--class_prompt=${cls_name} \
--val_prompt_path=${val_prompt_path} \
--local_out_path ${local_out_path} \
--data_path=${data_path} \
--exp_name=${exp_name}

