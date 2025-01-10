from transformers import ViTModel, ViTConfig
import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, hidden_dim=768, num_classes=4):

        super().__init__()
        vit_config = ViTConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=hidden_dim * 4,  # Feed-forward 层的大小
            patch_size=patch_size,
            image_size=img_size,
            hidden_act="gelu",
        )
        self.vit = ViTModel(vit_config)
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_classes = num_classes


        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, self.num_classes, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):

        B, C, H, W = x.shape  #  (B, 3, H, W)

        vit_output = self.vit(pixel_values=x).last_hidden_state  # (B, num_patches + 1, hidden_dim)

        # 去掉 [CLS] token，只保留 patch tokens
        patch_tokens = vit_output[:, 1:, :]  # (B, num_patches, hidden_dim)

        # 将 patch tokens 重塑为特征图
        h, w = H // self.patch_size, W // self.patch_size
        feature_map = patch_tokens.permute(0, 2, 1).contiguous().view(B, self.hidden_dim, h, w)

        # 解码为分割图像
        segmentation_map = self.decoder(feature_map)  # (B, num_classes, H, W)
        return segmentation_map


if __name__ == "__main__":

    model = ViT(img_size=224, patch_size=32, hidden_dim=768, num_classes=4)


    input_tensor = torch.randn(16, 3, 224, 224)
    output = model(input_tensor)

    print("输入尺寸:", input_tensor.shape)
    print("输出尺寸:", output.shape)