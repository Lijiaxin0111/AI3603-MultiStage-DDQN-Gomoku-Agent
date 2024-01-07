

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
import time 


# 获取命令行参数

thread_num = 2

episode = 19000
cnt = 82171

pis_root = r"D:\Program_File\piskvork\piskvork_win_lose"


ai_root = r"D:\Program_File\piskvork\other_agent"

ai2 = r"D:\Program_File\piskvork\other_agent\pbrain-wine.exe"

def run_bat_file(command):
    p = subprocess.Popen(command, shell=open)
    return p

ai_list = os.listdir(ai_root)


def run_bat_script(script_path, timeout,ai1,ai2,cnt):
    # 启动.bat脚本
    process = subprocess.Popen(script_path, shell=True)

    # 记录启动时间
    start_time = time.time()

    # 等待.exe程序启动
    time.sleep(1)

    # 检查.exe程序是否在运行
    while process.poll() is None:
        # 检查运行时间是否超过指定时间
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            # 超时，终止.exe程序
            subprocess.run("TASKKILL /T  /F /IM  "+ai1)
            subprocess.run("TASKKILL /T  /F /IM  "+ai2)
            subprocess.run("TASKKILL /T  /F /IM  "+ "piskvork.exe")

            process.terminate()
            print("[BREAK] KILL the exe")
            cnt[0] -= 1
            break
        else:
            time.sleep(1)



for i in range(episode):

    # 随机获取ai

    ai1 = os.path.join( random.sample(ai_list,k=1)[0])
    ai2 = os.path.join( random.sample(ai_list,k=1)[0])
    ai1 = os.path.join( random.sample(["pbrain-embryo18","pbrain-embryo21_e","pbrain-rapfi","pbrain-rapfi19","pbrain-rapfi21"],k=1)[0])
    print(ai1 + "  vs   " + ai2) 


    cnt += 1

    thread_list = []

    # 单线程
    out_file = "100_thousand_after\\out_better"+ str(cnt) +".txt"
    
    
    open_idx = str(random.randint(500,1000))

    
    parameters = [pis_root,ai1,ai2, out_file,open_idx,ai_root]
    command = "run_gomocu_manager.bat " + " ".join(parameters) 

    run_bat_script(command , 10,ai1, ai2, [cnt])


    # subprocess.run(command)



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



# pbrain-embryo18.exe  vs   pbrain-embryo18.exe
# outfile: data\out_better3087.txt
# opening_idx: 702
# pbrain-embryo18.exe  vs pbrain-embryo18.exe
# C:\Users\vipuser\Desktop\generate_data\piskvork\other_agent
# C:\Users\vipuser\Desktop\generate_data\piskvork\piskvork.exe -p C:\Users\vipuser\Desktop\generate_data\piskvork\other_agent\pbrain-embryo18.exe C:\Users\vipuser\Desktop\generate_data\piskvork\other_agent\pbrain-embryo18.exe -outfile data\out_better3087.txt -outfileformat 2 -opening 702