# Project 3 中期汇报

<a href='https://github.com/Lijiaxin0111/AI_3603_BIGHOME'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://notes.sjtu.edu.cn/dl9X8nY6TOSIFltbUP5y2g'><img src='https://img.shields.io/badge/MidtermReport-PDF-red'></a> <a href='https://huggingface.co/spaces/Gomoku-Zero/Demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'> </a>

#### 组长：李佳鑫
#### 组员：  黄奔皓、沈俊哲

### 一、 **文献调研：**
  - **AlphaGo家族:**
    - **AlphaGo在Gomuku的实现以及优化**： 《AlphaGomoku: An AlphaGo-based GomokuArtificial Intelligence using Curriculum Learning》
    - **alphaGo Zero**: 《Mastering the game of Go without human knowledge》
    - **Gumbel Muzero:**《POLICY IMPROVEMENT BY PLANNING WITH GUMBEL》
    - 《Accelerating Monte Carlo Tree Search with Probability Tree State Abstraction》
    - 《Switchable Lightweight Anti-Symmetric Processing (SLAP) with CNN Outspeeds Data Augmentation  by Smaller Sample– Application in Gomoku  Reinforcement Learning》
  - **网络方面的探索：** 
    - 《GomokuNet: A Novel UNet-style Network for Gomoku Zero Learning viaExploiting Positional Information and Multiscale Feature》




### 二、 **算法实现与优化：**

  - **完善对战与训练流程:**

    - 参考常见的网络训练框架，搭建模型训练框架，方便传入参数进行训练测试

    - 基于现有的对战流程，完善对战流程，能够允许各种模式下的测试训练和对战,包括：

      - `Alphazero(Dueling Net)`  VS  `Pure_MCST`

      -  `Gumbel_Alphazero(Dueling Net)`  VS  `Pure_MCST`

      - `Alphazero(Dueling Net)`  VS `Gumbel_Alphazero(Dueling Net)` 

        

    - 同时搭建**可视化IU**，进一步方便测试，参考demo网页: <a href='https://huggingface.co/spaces/Gomoku-Zero/Demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'> </a>，包括
    
      - `PureMCTS` vs `Player`
      - `AI aid` : `Alphazero` 辅助功能
      - 更多模型还有待导入
       </br>
       ![Demo](https://notes.sjtu.edu.cn/uploads/upload_2cc2de8e7cc0edbad00d0df0ddfae075.png)

​      

  - **算法实现**
    - 基于已有工作，对alphazero的网络框架进行调整，**实现DuelingNet 网络结构**
    - 基于《POLICY IMPROVEMENT BY PLANNING WITH GUMBEL》，实现**Gumbel_Alphazero**，实现较少仿真次数进而实现策略优化


- **性能测试**：**已经训练的模型**
  - `Alphazero` , `size` = 8 , `epochs` = 1500
  - `Alphazero` , `size` = 9 , `epochs` = 1500 + 1500
  - `Alphazero` , `size` = 12 , `epochs` = 1500 + 1500
  - `Alphazero` , `size` = 15 , `epochs` = 1500 + 1500
  - `Alphazero` + `Gumbel`, `size` = 9
  - `Alphazero` + `Dueling Net`, `size` = 9

    <center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://notes.sjtu.edu.cn/uploads/upload_1dce2030f06e02fc4b4850578edca711.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Alphazero在不同大小棋盘上的训练</div>
  </center>



### 三、 有待优化的问题：

- 在测试过程中发现，Agent对于边缘的棋子反应不敏感，当棋子落在边缘可能会导致Agent不能很好的进行很好的决策
    - 可能原因:
    
      AlphaZero的训练过程中使用了蒙特卡洛树搜索（Monte Carlo Tree Search）来探索游戏空间并评估不同的游戏状态。这种搜索方法通常会偏向于探索中央区域，因为边缘的棋子所能产生的合法移动较少，导致搜索树在边缘区域的展开受限。这可能导致边缘区域的棋局评估和决策相对较弱


- 在训练过程中，自我对弈生成训练数据的过程花销很大，考虑对训练过程进行优化
- 在Gumbel的仿真过程中，涉及不重复的仿真搜索，考虑使用多进程并行操作进行仿真，进一步提升Agent的速度

- 发现我们训练的alphazero决策价值网络容易过拟合，在输入一个棋盘状态过后，输出的策略分布经常是只有一个位置接近于1，其他位置趋近于0

    - 可能的原因:

         我们的训练过程是由自我对弈进行的，因此导致策略提升到一定程度，就不再又进一步上升空间，进而后面的棋盘状态保持相对固定，导致训练网络的过拟合，这也是边缘不敏感的重要原因
    


### 四、 技术路线：



- 基于已有问题，我们将从以下方面进行可能的探索优化:

    - 网络结构: 探索有可能的网络改进方法，对网络进行调优，进一步提升训练速度，同时提升网络对边缘的敏感度
        - 为网络输入增加Bias Layer，提高网络对棋盘输入中边缘点的鉴别能力
        <center>
        <img style="border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
        src="https://notes.sjtu.edu.cn/uploads/upload_49ce7f5f30224bdd0caec679af535c55.gif">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        padding: 2px;">通过为网络输入增加一个三维正态分布形状的bias(取负），来改进输入为单一的0或1值导致的丢失棋子在棋盘上的更具体的位置信息问题. Made With Manim</div>
            </center>
    
    - 
       并行处理： 探索对训练过程以及搜索过程中，重复进行的步骤，进行更进一步的并行处理优化
    
    
    - 数据增强: 针对过拟合的问题，尝试寻找更强代理对弈产生数据，优化训练过程或者搜集已有的五子棋对弈数据进行训练
