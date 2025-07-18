#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

GPU_ID=0
prompt="a <placeholder> dog holds a sign with the text 'ARBooth'"
model_path="./local_run/brown_cat/txt_emb-giter000K-ep300-iter2-last.pth"
save_file="./output.jpg"
seed=0

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 tools/run_arbooth.py \
    --prompt "${prompt}" --model_path ${model_path} --save_file ${save_file} --seed ${seed}

# ##############################################################################################
