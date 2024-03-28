# AI3603 GomokuAgent
[zh-CN](https://github.com/Lijiaxin0111/AI3603-MultiStage-DDQN-Gomoku-Agent/blob/main/README.zh_CN.md) | [en](https://github.com/Lijiaxin0111/AI3603-MultiStage-DDQN-Gomoku-Agent/blob/main/README.md)

<a href='https://github.com/Lijiaxin0111/AI_3603_BIGHOME'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://notes.sjtu.edu.cn/dl9X8nY6TOSIFltbUP5y2g'><img src='https://img.shields.io/badge/MidtermReport-PDF-red'></a> <a href='https://huggingface.co/spaces/Gomoku-Zero/Demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'> 





## 文件结构



`Gomoku_MCTS`:

- `cal_kill.py` 算杀模块
- `conv_utils.py`   把棋盘进行分块作为输入的尝试(经验证后效果不佳，并没有写入报告)
- `dueling_net.py`  DDQN 网络实现
- `game.py` 棋谱的定义、对弈的基本框架以及手动测试模型的代码
- `main_worker.py` 模型的训练框架
- `mcts_alphaZero.py` AlphaZero的实现
- `mcts_Gumbel_Muzero` Gumbel_MuZero的实现
- `mcts_pure.py` Pure MCST的代码
- `policy_value_net_pytorch_new.py` ？？？
- `policy_value_pytorch.py` 策略价值网络
- `scripts`: 测试代码的部分脚本
- `visualization`: 用于存放tensorboard的log
- `checkpoint`: 训练完成的模型参数
- `config`: 设置训练的基本参数


`Gomoku-Bot`:
* `test/` 测试脚本
* `board.py` 棋盘信息
* `cache.py` cache类
* `eval.py` 评估函数
* `gomoku_bot.py` 主文件，与 `Gomoku_MCTS` 接口对齐
* `minimax.py` alpha-beta 剪枝
* `position.py` 计算棋盘上棋子位置信息的一些函数
* `shape.py` 存储用于局势匹配计分的一些棋盘模式
* `zobrist.py` Zobrist 哈希实现





`generate_data`

- `text_files`: 内含用于生成数据的Gomoku manager 、 Gomuku agent、以及1000开局棋谱
- `crawl_new.py` : 跑取开局棋谱的脚本
- `generate_better_data.py`   、 `generate_better_data_avoid_down.py` 、`generate_data.py` : 利用Gomoku manager生成数据的脚本
- `Gomoku_MCTS_filter` ： 用于处理将数据按照胜负情况进行划分
- `run_gomocu_manager.bat` ： 用于生成数据的bat脚本，被`generate_data.py`调用运行

- `100_thousand_final`: 十万棋盘数据集







## 训练模型的环境配置与运行命令



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

其中的 `preload_model` 来自 Stage 1

### Stage 3: Competing with Masters

```sh
cd scripts

./third_stage_training.sh
```

其中的 `preload_model` 来自 Stage 2




## 生成数据代码的环境配置与运行命令

爬取“励精连珠教室”里面的比赛棋谱开局

```jsx
# 环境配置：第三方库安装
pip install beautifulsoup4
pip install selenium

# 环境配置：下载ChromeDriver，在代码中修改参数
# [ENV] 设置ChromeDriver的路径
chrome_driver_path = 'D:\\Program_File\\googledrive\\chromedriver.exe'
# 运行脚本前修改代码参数
# [CHANGE] 查看该比赛棋谱的网页数量修改下面的数字
	N_page = 5

# 目标网页的URL
# [CHANGE] 修改
  url = f'<https://www.ljrenju.com/news/cnjsu2022/zhesp2022r{i}.htm>'

# 运行命令
  python generate_data/crawler_new.py
```

设置Bot进行对弈产生数据

```jsx
# 运行命令
# 注意本代码因为涉及运行.bat脚本，只能在wins使用
# 如果在其他系统下运行，请将generate/run_gomocu_manager.bat转化为等价的bash文件
  python generate_data/generate_better_data.py
# 在github中的generate_data\\text_files已经存放了生成数据的exe文件
# 在其他地方运行参考下面的介绍
# 环境配置：下载Gomoku Manager， 并下载若干Bot
# 修改generate/generate_better_data.py 下面的参数

# [ENV] 在网站https://gomocup.org/download/, 下载piskvork.exe 并设置piskvork.exe路径
	pis_root = r"D:\\Program_File\\piskvork\\piskvork_win_lose"

# [ENV] 存放Bot的文件夹
	ai_root = r"D:\\Program_File\\piskvork\\other_agent"

#[CHANGE]存放输出数据的文件夹
  out_file = "100_thousand_after\\\\out_better"+ str(cnt) +".txt"
```

