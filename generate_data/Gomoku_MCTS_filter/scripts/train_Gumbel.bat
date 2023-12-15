@echo off



set train_options=--expri high_playout_recontinue_Gumbel_alphazero_train ^
--split train ^
--Player 1 ^
--batch_size 512 ^
--play_batch_size 1 ^
--board_width 9 ^
--board_height 9 ^
--use_gpu 0 ^
--mood 1  ^
--game_batch_num 1000 ^
--preload_model "C:\Users\li_jiaxin\Desktop\AI3603\BGWH\code\AI_3603_BIGHOME\Gomoku_MCTS\checkpoint\test_loss_2_Gumbel_alphazero_train_epochs=1000_size=9\best_policy.model" ^
--n_playout 500 ^



set run_cmd=python -u main_worker.py %train_options%

echo %run_cmd%
%run_cmd%