import torch
import torch.optim as optim
import torch.nn as nn
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
from torchio import SubjectsDataset, SubjectsLoader
import time
import csv  # 用于保存 Dice Loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import  torchio as tio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device = torch.device("cuda")  # 设置设备为 CUDA
    gpu_index = torch.cuda.current_device()  # 获取当前 GPU 索引
    gpu_name = torch.cuda.get_device_name(gpu_index)  # 获取 GPU 名称
    print(f"Using GPU: {gpu_name} (Index: {gpu_index})")
else:
    device = torch.device("cpu")  # 设置设备为 CPU
    print("CUDA is not available. Using CPU.")

def train_model(model, train_loader, val_loader, num_epochs, device, save_path,save_log_path):
    train_dice_losses = []  # 记录训练 Dice Loss
    val_dice_losses = []  # 记录验证 Dice Loss

    # 打开或创建 CSV 文件，写入表头
    os.makedirs(os.path.dirname(save_log_path), exist_ok=True)
    with open(save_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Dice_Loss', 'Val_Dice_Loss'])  # 写入表头

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_dice_loss = 0.0
        train_ce_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            logits = model(images)  # logits 的形状为 [B, C, H, W]

            # CrossEntropyLoss（多分类任务）
            ce_loss = cross_entropy_loss_fn(logits, masks)

            # DiceLoss（你的实现）
            dice_loss = dice_loss_fn(logits, masks, softmax=True)

            # 组合损失
            loss = 0.5 * ce_loss + 0.5 * dice_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_dice_loss += dice_loss.item()
            train_ce_loss += ce_loss.item()

        # 验证阶段
        model.eval()
        val_dice_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)
                # 根据需要进行 squeeze 或其他处理
                logits = model(images)
                dice_loss = dice_loss_fn(logits, masks, softmax=True)
                val_dice_loss += dice_loss.item()

        # 记录平均训练和验证 Dice Loss
        train_dice_losses.append(train_dice_loss / len(train_loader))
        val_dice_losses.append(val_dice_loss / len(val_loader))

        # 保存当前 epoch 的 Dice Loss 到 CSV 文件
        with open(save_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_dice_losses[-1], val_dice_losses[-1]])

        # 打印当前 epoch 的结果
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train CE Loss: {train_ce_loss / len(train_loader):.4f}, "
              f"Train Dice Loss: {train_dice_losses[-1]:.4f}, "
              f"Val Dice Loss: {val_dice_losses[-1]:.4f}, "
              f"Time: {epoch_time:.2f}s")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 绘制训练和验证 Dice Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_dice_losses, label="Train Dice Loss")
    plt.plot(range(1, num_epochs + 1), val_dice_losses, label="Val Dice Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Loss")
    plt.title("Dice Loss During Training")
    plt.legend()
    plt.savefig("dice_loss_curve.png")
    print("Training curve saved as 'dice_loss_curve.png'.")

# 验证阶段：计算每类 Dice Score
def evaluate_per_class_dice(model, data_loader, n_classes, device):
    model.eval()
    class_dice_scores = torch.zeros(n_classes, device=device)


    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images, masks = images.to(device), masks.to(device)
            # 根据需要进行 squeeze 或其他处理
            images = images.squeeze(-1).to(device)  # 去掉深度维度
            masks = masks.squeeze(-1).to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)  # Apply softmax for probabilities

            # 累计每个类的 Dice Score
            for i in range(n_classes):
                pred = probabilities[:, i]
                target = masks[:, i]
                smooth = 1e-5
                intersection = torch.sum(pred * target)
                union = torch.sum(pred) + torch.sum(target)
                dice = (2 * intersection + smooth) / (union + smooth)
                class_dice_scores[i] += dice

    class_dice_scores /= len(data_loader)  # 平均每类的 Dice Score
    return class_dice_scores.cpu().numpy()


training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(1.25),
    tio.Resize((256, 256, 1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneOf({
        tio.RandomAffine():0.2,
        tio.RandomElasticDeformation():0.2,
    }),

    tio.OneOf({
        tio.RandomBiasField():0.25,
        tio.RandomMotion():0.25,
        tio.RandomGhosting():0.25,
        tio.RandomSpike():0.25,
    }),
    tio.OneHot(num_classes=4)
])

validation_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(1.25),
    tio.Resize((256, 256, 1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneHot(num_classes=4)
])

test_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(1.25),
    tio.Resize((256, 256, 1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneHot(num_classes=4)
])


#--------------------------------------------------------------------------------------

learning_rate = 1e-3
num_epochs = 100
# Create data loaders
train_loader, val_loader, test_loader = create_data(
    ["OpenDatasets", "OpenDatasets"],
    '/Users/zhangzhe/PycharmProjects/data/OpenDataset',
    (256, 256),
    4,
    0.6,
    0.6,
    training_transform=training_transform,
    validation_transform=validation_transform,
    test_transform=test_transform,
)

num_classes = 4
input_channels = 1

# eg:save_path = "unet_new_model.pth"
# eg:save_log_path = "logs/unet_new_log.csv"
save_path = "unet_new_model.pth"
save_log_path = "logs/unet_new_log.csv"

model = UNet(in_channels=input_channels, num_classes=num_classes).to(device)
# model = ViT(img_size=512, patch_size=32, hidden_dim=768, num_classes=4)
# model = SETR(num_classes=4, image_size=512, patch_size=512//32, dim=1024, depth = 24, heads = 16, mlp_dim = 2048).to(device)

dice_loss_fn = DiceLoss(n_classes=num_classes)
cross_entropy_loss_fn = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


#--------------------------------------------------------------------------------------

train_model(model, train_loader, val_loader, num_epochs, device, save_path, save_log_path)

# 验证阶段：计算每类 Dice Score
class_dice_scores = evaluate_per_class_dice(model, test_loader, num_classes, device)
print(f"Dice Scores per Class: {class_dice_scores}")
print(f"LV (Class 1) Dice Score: {class_dice_scores[1]:.4f}")
print(f"MYO (Class 2) Dice Score: {class_dice_scores[2]:.4f}")
print(f"RV (Class 3) Dice Score: {class_dice_scores[3]:.4f}")