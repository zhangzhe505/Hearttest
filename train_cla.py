import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import create_data
from models.UNET import UNet
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
# 参数设置
learning_rate = 1e-3
num_epochs = 10
matrix_size = (512, 512)
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader, valid_loader, test_loader = create_data(
    ["OpenDatasets", "OpenDatasets"],
    '/Users/zhangzhe/PycharmProjects/data/OpenDataset',
    matrix_size= matrix_size,
    batch_size=batch_size,
    center_fraction=0.6,
    sampling_fraction=0.6,
)
# 初始化模型
num_classes = 2  # 二分类
input_channels = 1  # 输入图像的通道数
model = UNet(input_channels=input_channels, input_size=matrix_size, num_classes=num_classes).to(device)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 记录 loss 和 accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def calculate_accuracy(preds, labels):
    """
    计算准确率
    :param preds: 模型的预测结果 (B, H, W)
    :param labels: 真实标签 (B, H, W)
    :return: 准确率
    """
    preds = preds.argmax(dim=1)  # (B, H, W)，选择概率最大的类别
    correct = (preds == labels).float().sum()  # 计算预测正确的像素点数
    total = labels.numel()  # 总像素点数
    return correct / total

# 训练循环
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device).long()  # 确保 masks 为 long 类型


        # 前向传播
        outputs = model(images)
        print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录 loss 和准确率
        train_loss += loss.item()
        train_correct += (outputs.argmax(dim=1) == masks).sum().item()
        train_total += masks.numel()

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, masks, contains_class_1 in tqdm(valid_loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)

            # 跳过不包含类别 1 的样本
            if not contains_class_1.any():
                continue

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 记录 loss 和准确率
            val_loss += loss.item()
            val_correct += (outputs.argmax(dim=1) == masks).sum().item()
            val_total += masks.numel()

    val_loss /= len(valid_loader)
    val_accuracy = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), "unet_binary_segmentation.pth")

# 绘制损失和准确率曲线
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")  # 保存曲线图像
plt.show()