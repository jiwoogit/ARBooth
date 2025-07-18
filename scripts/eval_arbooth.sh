#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

GPU_ID=0
LOCAL_OUT=local_run
exp_name=brown_cat

val_prompt_path='prompts_live_objects.txt'
local_out_path=$LOCAL_OUT/${exp_name}

model_path=${LOCAL_OUT}
result_dir=local_results
mkdir -p $result_dir

############################### eval
# Generation samples for overall evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 tools/run_arbooth_for_eval.py \
    --result_dir ${result_dir} --model_path ${model_path}

cls_result_dir=local_pres_results
mkdir -p $cls_result_dir

# Generation samples for PRES metric evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 tools/run_arbooth_for_pres.py \
    --result_dir ${cls_result_dir} --model_path ${model_path}

# evaluation code
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 tools/evaluate.py \
    --result_dir ${result_dir} --cls_result_dir ${cls_result_dir}

##############################################################################################
