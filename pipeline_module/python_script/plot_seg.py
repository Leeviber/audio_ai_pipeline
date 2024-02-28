import numpy as np
import matplotlib.pyplot as plt

# 定义读取和绘制一维数组的函数
def plot_one_dim_array_from_binary(filename):
    array = np.fromfile(filename, dtype=np.int32)
    plt.plot(array, label='One-dimensional Array')
    plt.title('One-dimensional Array Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

# 读取二进制文件并转换为 numpy 数组
data = np.fromfile('binarized.bin', dtype=np.float64)
reshaped_data = data.reshape((9,293, 3))

# # 创建一个包含三个图表的 subplot
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# 第一个图表：画出每个说话者的均值数据
axs[0].set_title('Mean Data per Speaker')
axs[0].set_xlabel('Chunk')
axs[0].set_ylabel('Mean Value')

for i in range(3):
    axs[0].plot(reshaped_data[:, :, i].mean(axis=1), label=f'Speaker {i+1}')

axs[0].legend()

# 第二个图表：画出每个说话者的原始数据
axs[1].set_title('Raw Data per Speaker')
axs[1].set_xlabel('Chunk * Frame')
axs[1].set_ylabel('Value')

for i in range(3):
    axs[1].plot(reshaped_data[:, :, i].flatten(), label=f'Speaker {i+1}')

axs[1].legend()

# 第三个图表：画出一维数组数据
axs[2].set_title('One-dimensional Array Data')
axs[2].set_xlabel('Index')
axs[2].set_ylabel('Value')

# 调用绘制一维数组的函数并传入文件名
plot_one_dim_array_from_binary('count.bin')  # 替换成你的文件名

# 调整布局以避免重叠
plt.tight_layout()

# 显示图表
plt.show()
