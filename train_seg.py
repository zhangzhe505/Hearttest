import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datasets import create_data
from models.ViT import ViT
from models.SETR import SETR
from models.UNET_new import UNet
from loss.DiceLoss import DiceLoss
import time
from tqdm import tqdm
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if torch.cuda.is_available():
    device = torch.device("cuda")  # 设置设备为 CUDA
    gpu_index = torch.cuda.current_device()  # 获取当前 GPU 索引
    gpu_name = torch.cuda.get_device_name(gpu_index)  # 获取 GPU 名称
    print(f"Using GPU: {gpu_name} (Index: {gpu_index})")
else:
    device = torch.device("cpu")  # 设置设备为 CPU
    print("CUDA is not available. Using CPU.")

learning_rate = 1e-3
num_epochs = 1
# Create data loaders
train_loader, valid_loader, test_loader = create_data(
    ["OpenDatasets", "OpenDatasets"],
    '/Users/zhangzhe/PycharmProjects/data/OpenDataset',
    (512, 512),
    4,
    0.6,
    0.6,
)

num_classes = 4
input_channels = 1
model = UNet(in_channels=input_channels, num_classes=num_classes).to(device)
# model = ViT(img_size=512, patch_size=32, hidden_dim=768, num_classes=4)
# model = SETR(num_classes=4, image_size=512, patch_size=512//32, dim=1024, depth = 24, heads = 16, mlp_dim = 2048).to(device)

# 损失函数和优化器
criterion = DiceLoss(num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 存储损失和 IoU
train_losses = []
val_losses = []
train_ious = []
val_ious = []

# 定义辅助损失的权重
loss_weights = [0.4, 0.3, 0.2, 0.1]  # 权重和为1

# 训练和验证
for epoch in range(num_epochs):
    start_time = time.time()

    # 训练阶段
    model.train()
    train_loss = 0
    train_iou = torch.zeros(num_classes - 1, device=device)  # 存储每个类别的 IoU
    print(f"\nEpoch [{epoch + 1}/{num_epochs}] Training:")
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # SETR 会返回多层输出

        # 计算损失
        total_loss = 0
        for i, output in enumerate(outputs):  # 遍历每一层输出
            loss = criterion(output, masks)
            total_loss += loss_weights[i] * loss  # 按权重累加损失

        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

        # 计算 IoU
        preds = torch.argmax(outputs[-1], dim=1)  # 使用最后一层输出作为预测结果
        for cls in range(1, num_classes):  # 计算每个类的 IoU（忽略背景类）
            intersection = torch.sum((preds == cls) & (masks == cls))
            union = torch.sum((preds == cls) | (masks == cls))
            train_iou[cls - 1] += intersection / (union + 1e-6)

    train_loss /= len(train_loader)
    train_iou /= len(train_loader)

    # 验证阶段
    model.eval()
    val_loss = 0
    val_iou = torch.zeros(num_classes - 1, device=device)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Validating:")
    with torch.no_grad():
        for images, masks in tqdm(valid_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)  # SETR 会返回多层输出

            # 计算损失
            total_loss = 0
            for i, output in enumerate(outputs):  # 遍历每一层输出
                loss = criterion(output, masks)
                total_loss += loss_weights[i] * loss  # 按权重累加损失

            val_loss += total_loss.item()

            # 计算 IoU
            preds = torch.argmax(outputs[-1], dim=1)  # 使用最后一层输出作为预测结果
            for cls in range(1, num_classes):  # 计算每个类的 IoU（忽略背景类）
                intersection = torch.sum((preds == cls) & (masks == cls))
                union = torch.sum((preds == cls) | (masks == cls))
                val_iou[cls - 1] += intersection / (union + 1e-6)

    val_loss /= len(valid_loader)
    val_iou /= len(valid_loader)

    # 打印每个 epoch 的结果
    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train IoU (LV/MYO/RV): {train_iou.cpu().numpy()}, "
          f"Val Loss: {val_loss:.4f}, Val IoU (LV/MYO/RV): {val_iou.cpu().numpy()}, "
          f"Time: {epoch_time:.2f}s")

    # 保存每个 epoch 的损失和 IoU
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_ious.append(train_iou.cpu().numpy())
    val_ious.append(val_iou.cpu().numpy())

# 保存模型
torch.save(model.state_dict(), "SETR_trained.pth")

# 绘制 Loss 曲线
epochs = np.arange(1, num_epochs + 1)
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SETR Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# 绘制 IoU 曲线
plt.figure(figsize=(8, 6))
iou_classes = ['LV', 'MYO', 'RV']
for cls in range(num_classes - 1):  # 遍历每个类别
    plt.plot(epochs, [iou[cls] for iou in train_ious], label=f"Train {iou_classes[cls]} IoU")
    plt.plot(epochs, [iou[cls] for iou in val_ious], label=f"Val {iou_classes[cls]} IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("SETR IoU Curve")
plt.legend()
plt.tight_layout()
plt.savefig("SETR_iou_curve.png")
plt.show()