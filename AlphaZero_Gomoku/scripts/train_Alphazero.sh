#!/bin/bash
script_dir=$(cd $(dirname $0);pwd)
dir=$(dirname $script_dir)
train_options="
--board_width 9 \
--board_height 9 \
--biased True \
--n_playout 1000 \
--preload_model ./checkpoint/epochs=3000_size=9_biased=True/best_policy.model"

#
#train_options="
#--board_width 9 \
#--board_height 9 \
#--biased True \
#--n_playout 3000 \
#--preload_model /Users/husky/AI_3603_BIGHOME/AlphaZero_Gomoku/checkpoint/epochs=1500_size=9_training2/best_policy.model"
#

run_cmd="python -u $dir/train.py $train_options"
echo $run_cmd
$run_cmd
