import numpy as np
import matplotlib.pyplot as plt
import os
'二维numpy数据可视化'

def display_numpy_as_image(file_path, cmap='gray'):
    """
    加载指定路径的二维 NumPy 数组并显示为图片。

    :param file_path: str, .npy 文件的路径
    :param cmap: str, 显示图像的颜色映射表（默认灰度图 'gray'）
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    try:
        # 加载 .npy 文件
        data = np.load(file_path)
    except Exception as e:
        print(f"无法加载文件 {file_path}：{e}")
        return

    # 检查数据维度
    if len(data.shape) != 2:
        print(f"错误：文件中的数据不是二维数组，实际维度为 {data.shape}")
        return
    print(data.reshape)
    # 显示图像
    plt.imshow(data, cmap=cmap)
    plt.colorbar()  # 显示颜色条
    plt.title(f"Image from {os.path.basename(file_path)}")
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    unique_classes = np.unique(data)  # 获取所有唯一的类别值
    print(f"类别数：{len(unique_classes)}")  # 类别数量
    print(f"类别值：{unique_classes}")  # 类

# 示例调用
if __name__ == "__main__":
    file_path = "/Users/zhangzhe/PycharmProjects/data/OpenDataset/val/A5C2D2_slice1_ed24_gt.npy"
    display_numpy_as_image(file_path)