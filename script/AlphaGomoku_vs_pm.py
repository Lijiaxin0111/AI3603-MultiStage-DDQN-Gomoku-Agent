import subprocess

# 启动一个CMD终端
cmd_process = subprocess.Popen(['cmd.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 向CMD发送命令
cmd_command = b"echo 'Hello from CMD'\n"
cmd_process.stdin.write(cmd_command)
cmd_process.stdin.flush()

# 读取CMD的输出
output = cmd_process.stdout.read()
print("CMD Output:", output)

# 关闭CMD终端
cmd_process.stdin.write("exit\n")
cmd_process.stdin.flush()

cmd_process.wait()

