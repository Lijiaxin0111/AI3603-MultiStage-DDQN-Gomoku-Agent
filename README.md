# README





## 文件结构



`generate_data`

- `text_files`: 内含用于生成数据的Gomoku manager 、 Gomuku agent、以及1000开局棋谱
- `crawl_new.py` : 跑取开局棋谱的脚本
- `generate_better_data.py`   、 `generate_better_data_avoid_down.py` 、`generate_data.py` : 利用Gomoku manager生成数据的脚本
- `Gomoku_MCTS_filter` ： 用于处理将数据按照胜负情况进行划分
- `run_gomocu_manager.bat` ： 用于生成数据的bat脚本，被`generate_data.py`调用运行

- `100_thousand_final`: 十万棋盘数据集



``

【TODO】



## 训练模型的环境配置与运行命令



【TODO】





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