import numpy as np
import matplotlib.pyplot as plt

# 读取二进制文件并转换为numpy数组
data = np.fromfile('data.bin', dtype=np.float64)
data = data.reshape((113, 293, 3))
line1 = data[:, :, 0]

x = np.arange(113)
plt.plot(x, data[:,:,1], label='Line 1')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
# # 绘制折线图
# x = np.arange(293)  # x轴坐标
# plt.plot(x, data[:, :, 0], label='Line 1')
# plt.plot(x, data[:, :, 1], label='Line 2')
# plt.plot(x, data[:, :, 2], label='Line 3')

# # 添加图例和标签
# plt.legend()
# plt.xlabel('X')
# 

# 显示图形
plt.show()
