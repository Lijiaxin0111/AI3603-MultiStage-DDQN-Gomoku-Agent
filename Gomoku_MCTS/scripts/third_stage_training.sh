#!/bin/bash

# 获取脚本所在目录的绝对路径
script_dir=$(dirname "$(readlink -f "$0")")

# 进入父目录
cd "$script_dir/.."

export train_options="--expri third-stage-duel \
--split train \
--Player 0 \
--model_type duel \
--batch_size 512 \
--play_batch_size 1 \
--board_width 9 \
--board_height 9 \
--use_gpu 0 \
--mood 0 \
--game_batch_num 5 \
--pure_mcts_playout_num 2000 \
--n_game 20 \
--data_collect 1 \
--check_freq 1 \
--start_edge
--c_puct2 0 \
--high_player gomokubot \
--preload_model ./checkpoint/first-stage-100th-duel_epochs=1000_size=9_model=duel/best_policy.model
"

run_cmd="python -u main_worker.py $train_options"

$run_cmd