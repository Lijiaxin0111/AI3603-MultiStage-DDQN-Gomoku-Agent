"""
FileName: generate
Author: Jiaxin Li
Create Date: 2023/11/21
Description: 生成对弈数据:单线程 多线程（有点卡）
Edit History:
Debug: the dim of output: probs
"""



import subprocess
import sys
import random
import os


# 获取命令行参数

thread_num = 2


episode = 25000
cnt = 83635

# 差一千五

# [ENV] 在网站https://gomocup.org/download/, 下载piskvork.exe 并设置piskvork.exe路径
pis_root = r"generate_data\text_files\manager"

# [ENV] 存放Bot的文件夹
ai_root = r"generate_data\text_files\bots_pool"

# ai2 = r"D:\Program_File\piskvork\other_agent\pbrain-wine.exe"

def run_bat_file(command):
    p = subprocess.Popen(command, shell=open)
    return p

ai_list = os.listdir(ai_root)


for i in range(episode):
    # 随机获取ai

    #[ENV] 设置Bot名称
    ai2 = os.path.join( random.sample(ai_list,k=1)[0])
    ai1 = os.path.join( random.sample(["pbrain-SlowRenju.exe","pbrain-SlowRenju_x64.exe","pbrain-whose20190401x64.exe","pbrain-Yixin2018.exe","pbrain-pela.exe"],k=1)[0])
    
    # ai1 = "pbrain-embryo18.exe"
    # ai2 = "pbrain-embryo18.exe"
    print(ai1 + "  vs   " + ai2) 

    cnt += 1

    thread_list = []

    #[CHANGE]存放输出数据的文件夹
    out_file = r"generate_data\text_files\outfiles\out_better"+ str(cnt) +".txt"
    
    open_idx = str(random.randint(500,1000))
    
    parameters = [pis_root,ai1,ai2, out_file,open_idx,ai_root]
    # print(parameters)
    command = r"generate_data\run_gomocu_manager.bat " + " ".join(parameters) 


    subprocess.run(command)



#多线程会导致很多的系统中断，让电脑超级卡
    # for j in range(thread_num):
    #     cnt += 1    
    #     out_file = r"data\out"+ str(cnt) +".txt"
    #     open_idx = str(random.randint(1,40))
    #     parameters = [pis_root,ai1,ai2, out_file,open_idx]
    #     command = "run_gomocu_manager.bat " + " ".join(parameters) 

    #     thread_list.append(run_bat_file(command))
    
    # for pj in thread_list :
    #     pj.wait()



