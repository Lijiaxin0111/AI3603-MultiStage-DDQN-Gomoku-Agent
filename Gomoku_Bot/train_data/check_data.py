import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def load_data():
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

path = 'data/train_data.pkl'
data = load_data()
print(len(data[1]))
exit(0)
scores = data['scores']
state = data['state']

# 创建一个空的15x15地图
map = np.zeros((15, 15))

# 在地图上根据state设置黑白棋子的位置
for i in range(15):
    for j in range(15):
        if state[i][j] == 1:
            map[i][j] = -1  # 黑色棋子
        elif state[i][j] == -1:
            map[i][j] = 1  # 白色棋子

# 绘制地图
fig, ax = plt.subplots()
im = ax.imshow(map, cmap='gray')

# 使用scores在地图上注释颜色
for i in range(15):
    for j in range(15):
        text = ax.text(j, i, scores[i][j], ha='center', va='center',
                       color='black' if map[i][j] == 1 else 'white')

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 设置标题
ax.set_title('Map with Scores')

plt.show()