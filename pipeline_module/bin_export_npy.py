import os
import numpy as np

def read_bin_files(folder_path):
    files = sorted([file for file in os.listdir(folder_path) if file.endswith('.bin')])

    arrays = []
    i=0
    for file in files:

        file_path = os.path.join(folder_path, file)
        array = np.fromfile(file_path, dtype=np.float32).reshape(1, 198, 80)
        filename = f"feat_npy/{i+1:03d}.npy"
        np.save(filename, array)
        i+=1
    return len(files)

 
folder_path = 'feature_bin'  # 修改为包含.bin文件的文件夹路径
output_path = 'output.npy'  # 修改为保存.npy文件的路径和文件名

num = read_bin_files(folder_path)
print(num)
# 保存文件名到txt
with open("dataset.txt", "w") as file:
    for i in range(num):
        filename = f"input_npy/{i+1:03d}.npy"
        file.write(filename + "\n")