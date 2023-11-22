#!/bin/bash
script_dir=$(cd $(dirname $0);pwd)
dir=$(dirname $script_dir)
train_options="
--play_batch_size 1 \
--board_width 12 \
--board_height 12 \
--preload_model /Users/husky/AI_3603_BIGHOME/AlphaZero_Gomoku/checkpoint/epochs=1000_size=12_training1/best_policy.model"

run_cmd="python -u $dir/train.py $train_options"
echo $run_cmd
$run_cmd
