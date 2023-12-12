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

episode = 200
cnt = 250


pis_root = r"D:\Program_File\piskvork\piskvork_win_lose"


ai_root = r"D:\Program_File\piskvork\other_agent"

ai2 = r"D:\Program_File\piskvork\other_agent\pbrain-wine.exe"

def run_bat_file(command):
    p = subprocess.Popen(command, shell=open)
    return p

ai_list = os.listdir(ai_root)



for i in range(episode):

    # 随机获取ai

    ai1 = os.path.join(ai_root, random.sample(ai_list,k=1)[0])
    ai2 = os.path.join(ai_root, random.sample(ai_list,k=1)[0])
    print(ai1 + "  vs   " + ai2) 

    cnt += 1

    thread_list = []

    # 单线程
    out_file = "data\\out_better"+ str(cnt) +".txt"
    
    
    open_idx = str(random.randint(1,60))
    parameters = [pis_root,ai1,ai2, out_file,open_idx]
    command = "run_gomocu_manager.bat " + " ".join(parameters) 


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



