# AI3603 Final Project: Gomoku Agent

<a href='https://github.com/Lijiaxin0111/AI_3603_BIGHOME'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://notes.sjtu.edu.cn/dl9X8nY6TOSIFltbUP5y2g'><img src='https://img.shields.io/badge/MidtermReport-PDF-red'></a> <a href='https://huggingface.co/spaces/Gomoku-Zero/Demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'> 

## Todo
- [ ] **Huggingface Space Web 多模型部署**
  - [ ] Human VS AI
    - [x] AI assistant
  - [ ] AI VS AI
- [ ] **Alphazero边缘注意力提升**
    - [ ] Board + Gaussian Distribution
    - [ ] Let Alphazero play with huamn to collect datas relevant to edge-policy improvement?
- [ ] Midterm Report
- [ ] Final Report



### 任务进度记录：

- 2023-11-21.ljx.
  - 复现了Gumbel_MCST
  - 在main_workder里面增加了"--Player" 参数,'0' 为 Alphazero, '1' 为 Gumbel Alphazero 
  - 在main_workder里面增加了"--mood" 参数,0: Alphazero Vs Pure;  1: Gumbel_Alphazero Vs Pure; 2:Alphazero Vs Gumbel_Alphazero ,仅在test，valid的时候起作用, train的时候记得设置好！！！不然valid的时候就不是需要的那个模型

- 2023-11-19.hbh.
  - 完成了可视化界面的初稿
  
- 2023-11-18.hbh.
  - 跟进代码，了解项目结构
  - 调研了一些最新的论文，了解优化方法
  - 复习AlphaZero的原理

- 2023-11-18.sjz.
  -  将PI2.0服务器上的conda环境配置完成
  -  对收集自我对弈数据部分进行了多进程并行，对playout部分进行了多线程并行

- 2023-11-17.sjz.
  -  将思源一号服务器上的conda环境配置完成
  -  对网络进行改进： 使用DuelingNet的架构: 有待验证 性能

- 2023-11-17. ljx. 对训练框架进行简单修改，增加了：
  - 将网络里面train_step 的代码移植到main_worker中,之后写value net可以不用写train step函数

- 2023-11-16. ljx. 对训练框架进行简单修改，增加了：
  - 训练参数和代码的分离，可以通过命令行 、编写scrpt.sh （.bat） 脚本直接调整训练参数，方便后面写demo训练脚本
  - 增加了进度条显示，不过因为他们的小训练轮次以及采集数据的轮次有点短，效果还不是很好
  - 增加了用tensorboard 可视化曲线的功能，可视化”loss“，win_ratio 等
