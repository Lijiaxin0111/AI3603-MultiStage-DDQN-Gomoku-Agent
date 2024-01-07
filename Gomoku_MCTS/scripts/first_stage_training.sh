#!/bin/bash

# 获取脚本所在目录的绝对路径
script_dir=$(dirname "$(readlink -f "$0")")

# 进入父目录
cd "$script_dir/.."

export train_options="--expri final-100th-duel \
--split train \
--Player 0 \
--model_type duel \
--batch_size 512 \
--play_batch_size 1 \
--board_width 9 \
--board_height 9 \
--use_gpu 0 \
--mood 0 \
--data_collect 2 \
--check_freq 25 \
"

run_cmd="python -u main_worker.py $train_options"

$run_cmd