# AI3603 GomokuAgent

[zh-CN](https://github.com/Lijiaxin0111/AI3603-MultiStage-DDQN-Gomoku-Agent/blob/main/README.zh_CN.md) | [en](https://github.com/Lijiaxin0111/AI3603-MultiStage-DDQN-Gomoku-Agent/blob/main/README.md)

<a href='https://github.com/Lijiaxin0111/AI_3603_BIGHOME'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://notes.sjtu.edu.cn/dl9X8nY6TOSIFltbUP5y2g'><img src='https://img.shields.io/badge/MidtermReport-PDF-red'></a> <a href='https://huggingface.co/spaces/Gomoku-Zero/Demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'> 



## File Structure

`Gomoku_MCTS`:
- `cal_kill.py` - Kill calculation module
- `conv_utils.py` - Attempt to chunk the board for input (proven ineffective, not included in the report)
- `dueling_net.py` - DDQN network implementation
- `game.py` - Game record definition, basic gameplay framework, and manual testing code
- `main_worker.py` - Model training framework
- `mcts_alphaZero.py` - AlphaZero implementation
- `mcts_Gumbel_Muzero` - Gumbel_MuZero implementation
- `mcts_pure.py` - Pure MCST code
- `policy_value_net_pytorch_new.py` - Policy value network (unclear purpose)
- `policy_value_pytorch.py` - Policy value network
- `scripts`: Scripts for testing code
- `visualization`: Folder for tensorboard logs
- `checkpoint`: Trained model parameters
- `config`: Basic training configuration

`Gomoku-Bot`:
- `test/` - Testing scripts
- `board.py` - Board information
- `cache.py` - Cache class
- `eval.py` - Evaluation function
- `gomoku_bot.py` - Main file, aligned with `Gomoku_MCTS` interface
- `minimax.py` - Alpha-beta pruning implementation
- `position.py` - Functions for computing board position information
- `shape.py` - Patterns for scoring based on board situations
- `zobrist.py` - Zobrist hashing implementation

`generate_data`:
- `text_files`: Contains Gomoku manager and agent scripts and 1000 opening game records
- `crawl_new.py`: Script for scraping opening game records
- `generate_better_data.py`, `generate_better_data_avoid_down.py`, `generate_data.py`: Scripts for generating data with Gomoku manager
- `Gomoku_MCTS_filter`: For sorting data based on win/loss
- `run_gomocu_manager.bat`: Batch script for data generation, invoked by `generate_data.py`
- `100_thousand_final`: Dataset of 100,000 board positions

## Model Training Setup and Commands

### Stage 1: Imitation Learning
```sh
cd scripts
./first_stage_training.sh
```

### Stage 2: Self-Play Training
```sh
cd scripts
./second_stage_self_training.sh
```
`preload_model` should be from Stage 1.

### Stage 3: Competing with Masters
```sh
cd scripts
./third_stage_training.sh
```
`preload_model` should be from Stage 2.

## Data Generation Setup and Commands

To scrape opening game records from "Lijin Continuum Classroom":
```jsx
# Setup: Install third-party libraries
pip install beautifulsoup4
pip install selenium

# Setup: Download ChromeDriver and modify the path in the code
chrome_driver_path = 'D:\\Program_File\\googledrive\\chromedriver.exe'
# Modify the following number based on the webpage count of the game records
N_page = 5

# Target URL (modify accordingly)
url = f'<https://www.ljrenju.com/news/cnjsu2022/zhesp2022r{i}.htm>'

# Run command
python generate_data/crawler_new.py
```

Setting up the Bot for gameplay to generate data:
```jsx
# Run command
python generate_data/generate_better_data.py
# Note: This script involves running a .bat file, so it only works on Windows.
# For other systems, convert the generate/run_gomocu_manager.bat to an equivalent bash file.

# Setup: Download Gomoku Manager and several Bots
# Modify the parameters in generate/generate_better_data.py

# [ENV] Download piskvork.exe from https://gomocup.org/download/ and set the path
pis_root = r"D:\\Program_File\\piskvork\\piskvork_win_lose"

# [ENV] Folder containing the Bots
ai_root = r"D:\\Program_File\\piskvork\\other_agent"

# [CHANGE] Folder for saving the output data
out_file = "100_thousand_after\\\\out_better" + str(cnt) + ".txt"
```
