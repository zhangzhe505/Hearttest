import nibabel as nib
import numpy as np
import os

files_path = '/Users/zhangzhe/PycharmProjects/data/OpenDataset/train/A0S9V9_slice3_es9.npy'
data = nib.load(files_path)
print(f"data shape:{data.shape}")
# unique_classes = np.unique(data)  # 获取所有唯一的类别值
# print(f"类别数：{len(unique_classes)}")  # 类别数量
# print(f"类别值：{unique_classes}")  # 类

header = data.header
voxel_size = header.get_zooms()
print(f"体素大小: {voxel_size}")