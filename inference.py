import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms.functional import to_pil_image


def visualize_segmentation(model, weights_path, image_path, class_colors, device='cpu'):

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)

    # 加载 .npy 格式的图像
    image = np.load(image_path)
    image = np.expand_dims(image, axis=0) #(C, H, W)
    # 在加载数据时，将灰度图扩展为 RGB
    if len(image.shape) == 2:  # 如果是灰度图
        image = np.expand_dims(image, axis=0)  # 扩展为 (1, H, W)
    image = np.repeat(image, 3, axis=0)  # 复制 3 次，扩展为 (3, H, W)

    image = torch.from_numpy(image).float().unsqueeze(0)  #(1, C, H, W)
    image = image.to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(image)  # 模型输出 (B, num_classes, H, W)
        pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()  # 取类别

    # 生成叠加图像
    original_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 转为 (H, W, C)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255
    original_image = original_image.astype(np.uint8)
    # if original_image.max() > 1:  # 如果原始图像范围是 [0, 255]，归一化到 [0, 1]
    #     original_image = original_image / 255.0
    # original_image = (original_image).astype(np.uint8)  # 转为 uint8 类型
    #
    # 确保 original_image 是三通道 RGB
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:  # 如果是灰度图
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)  # 转为 RGB
    # 确保 overlay 是三通道
    H, W = original_image.shape[:2]
    overlay = np.zeros((H, W, 3), dtype=np.uint8)  # 初始化为 RGB 图像

    for cls_idx, color in enumerate(class_colors, start=1):
        # 避免广播错误，逐通道赋值
        mask = (pred == cls_idx)  # 获取当前类别的掩码
        for c in range(3):  # 逐通道赋值
            overlay[..., c][mask] = color[c]

    # 调整 overlay 的大小以匹配 original_image
    overlay = cv2.resize(overlay, (original_image.shape[1], original_image.shape[0]))

    # 混合原图与分割结果
    blended = cv2.addWeighted(original_image, 0.6, overlay, 0.4, 0)

    # 显示图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray' if original_image.shape[-1] == 1 else None)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Segmentation")
    plt.imshow(pred, cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(blended)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
# 示例调用
if __name__ == "__main__":
    from models.UNET_SEG import UNet
    from models.ViT import ViT
    from models.SETR import SETR

    num_classes = 4
    # model = UNet(input_channels=1, num_classes=num_classes)
    model = ViT(img_size=512, patch_size=32, hidden_dim=768, num_classes=4)
    # model = SETR(num_classes=32, image_size=512, patch_size=512//32, dim=1024, depth = 24, heads = 16, mlp_dim = 2048)

    weights_path = "/Users/zhangzhe/PycharmProjects/gitproject/Hearttest/ViT_segmentation.pth"

    # 设置输入图像路径
    image_path = "/Users/zhangzhe/PycharmProjects/data/OpenDataset/train/A0S9V9_slice3_es9.npy"
    data = np.load(image_path)
    print("像素亮度范围: [{}, {}]".format(data.min(), data.max()))
    class_colors = [
        (255, 0, 0),   # 类别 1 -> 红色
        (0, 255, 0),   # 类别 2 -> 绿色
        (0, 0, 255)    # 类别 3 -> 蓝色
    ]

    # 运行分割
    visualize_segmentation(model, weights_path, image_path, class_colors, device='cpu')