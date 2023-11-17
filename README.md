# AI_3603_BIGHOMEWORK



### 任务进度记录：

- 2023-11-16. ljx. 对训练框架进行简单修改，增加了：
  - 训练参数和代码的分离，可以通过命令行 、编写scrpt.sh （.bat） 脚本直接调整训练参数，方便后面写demo训练脚本
  - 增加了进度条显示，不过因为他们的小训练轮次以及采集数据的轮次有点短，效果还不是很好
  - 增加了用tensorboard 可视化曲线的功能，可视化”loss“，win_ratio 等


- 2023-11-17. ljx. 对训练框架进行简单修改，增加了：
  - 将网络里面train_step 的代码移植到main_worker中,之后写value net可以不用写train step函数

- 2023-11-17.sjz. 将sylogin服务器上的conda环境配置完成


