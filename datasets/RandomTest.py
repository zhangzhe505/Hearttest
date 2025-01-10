import torch
from torch.utils.data import Dataset
import numpy as np
import random


class RandomDataset(Dataset):
    def __init__(self, num_samples, image_size=(1, 256, 256)):
        """
        三分类任务的数据集，具有固定随机种子，确保每次运行生成的样本一致。

        参数:
        - num_samples: 数据集中样本数量。
        - image_size: 图像的大小，默认为单通道 (1, 256, 256)。
        """
        seed = 100
        np.random.seed(seed)
        torch.manual_seed(seed)


        self.num_samples = num_samples
        self.image_size = image_size
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            # 初始化图像为全 0
            image = torch.zeros(*image_size)
            # 随机生成一个类别标签 (0, 1, 2)
            label = random.randint(0, 2)

            # 根据类别生成规则化的图像模式
            height, width = self.image_size[1], self.image_size[2]

            if label == 0:
                # 类别 0：中心区域全白
                offset_h = random.randint(-10, 10)
                offset_w = random.randint(-10, 10)
                # 确保偏移量在图片范围之内
                h_start = max(height // 4 + offset_h, 0)
                h_end = min(3 * height // 4 + offset_h, height)
                w_start = max(width // 4 + offset_w, 0)
                w_end = min(3 * width // 4 + offset_w, width)
                image[:, h_start:h_end, w_start:w_end] = 1

            elif label == 1:
                # 类别 1：左上角全白
                scale = random.uniform(0.8, 1.2)
                h_size = int((height // 2) * scale)
                w_size = int((width // 2) * scale)
                h_start, h_end = 0, min(h_size, height)
                w_start, w_end = 0, min(w_size, width)
                image[:, h_start:h_end, w_start:w_end] = 1

            elif label == 2:
                # 类别 2：右下角全白
                scale = random.uniform(0.8, 1.2)
                h_size = int((height // 2) * scale)
                w_size = int((width // 2) * scale)
                h_start, h_end = max(height - h_size, 0), height
                w_start, w_end = max(width - w_size, 0), width
                image[:, h_start:h_end, w_start:w_end] = 1

            # 添加随机噪声
            noise = torch.rand_like(image) * 0.1
            image += noise
            image = torch.clamp(image, 0, 1)

            # 将图像和对应的标签存入列表
            self.data.append(image)
            self.labels.append(torch.tensor(label))  # 每个像素都是类别标签

        # 将图像和标签转换为 PyTorch 张量
        self.data = torch.stack(self.data)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # 生成数据集
    dataset = RandomDataset(num_samples=5, image_size=(1, 128, 128))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 可视化一个样本
    for images, labels in data_loader:
        image = images[0].squeeze(0).numpy()  # 去掉通道维度
        label = labels[0].numpy()  # 标签矩阵

        plt.figure(figsize=(10, 5))

        # 显示图像
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image, cmap="gray")
        plt.colorbar()

        # 显示标签
        plt.subplot(1, 2, 2)
        plt.title("Label")
        plt.imshow(label, cmap="viridis")
        plt.colorbar()

        plt.show()
        break