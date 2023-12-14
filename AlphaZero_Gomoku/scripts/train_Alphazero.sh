#!/bin/bash
script_dir=$(cd $(dirname $0);pwd)
dir=$(dirname $script_dir)

#train_options="
#--board_width 9 \
#--board_height 9 \
#--biased True \
#--n_playout 500 \
#--preload_model ./checkpoint/epochs=3000_size=9_biased=True/best_policy.model"

#train_options="
#--board_width 9 \
#--board_height 9 \
#--biased True \
#--n_playout 500 \
#--temp 5 \
#"


#
train_options="
--board_width 9 \
--board_height 9 \
--n_playout 3000 \
--temp 0.7 \
--preload_model /Users/husky/AI_3603_BIGHOME/AlphaZero_Gomoku/checkpoint/2023-11-27_11-09-22_epochs=3000_size=9_biased=False_simultime=3000_temp=0.7_exp=False_net=False/current_policy.model"


#train_options="
#--board_width 9 \
#--board_height 9 \
#--n_playout 500 \
#--temp 0.7 \
#--new_net True \
#--preload_model /Users/husky/AI_3603_BIGHOME/AlphaZero_Gomoku/checkpoint/2023-11-27_09-33-39_epochs=3000_size=9_biased=False_simultime=500_temp=0.7_exp=False_net=True/current_policy.model
#"


run_cmd="python -u $dir/train.py $train_options"
echo $run_cmd
$run_cmd
