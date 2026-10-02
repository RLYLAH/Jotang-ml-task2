"""
简单 CNN：4 个卷积块 + 全局平均池化 + 1 个输出神经元
输入 3x128x128，输出 1 个 logit（猫=0，狗=1）
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                # 尺寸减半
            )

        self.features = nn.Sequential(
            block(3,   32),    # 128 -> 64
            block(32,  64),    # 64  -> 32
            block(64,  128),   # 32  -> 16
            block(128, 256),   # 16  -> 8
        )

        self.gap = nn.AdaptiveAvgPool2d(1)     # 8x8 -> 1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # 打印每层输出形状
    model = SimpleCNN()
    x = torch.randn(2, 3, 128, 128)
    print(f"输入: {tuple(x.shape)}")
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"features[{i}] 后: {tuple(x.shape)}")
    x = model.gap(x)
    print(f"GAP 后: {tuple(x.shape)}")
    x = model.classifier(x)
    print(f"输出: {tuple(x.shape)}")