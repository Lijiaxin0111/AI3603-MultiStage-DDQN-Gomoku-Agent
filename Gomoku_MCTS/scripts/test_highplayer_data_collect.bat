@echo off



set train_options=--expri nrm_test_res_num_7 ^
--split train ^
--mood 0  ^
--batch_size 512 ^
--play_batch_size 128 ^
--board_width 9 ^
--board_height 9 ^
--use_gpu 0 ^
--data_collect 2 ^
--check_freq 100 ^
--data_augment 50 ^
--model_type normal ^
--game_batch_num 2000 ^
--shown 1

set run_cmd=python -u main_worker.py %train_options%

echo %run_cmd%
%run_cmd%