import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 定义MBConv块
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(Swish())

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.skip_add = nn.Identity()

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

# 定义整个网络架构
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        self.stage1 = MBConv(32, 16, 3, 1, 1)
        self.stage2 = nn.Sequential(
            MBConv(16, 24, 3, 2, 6),
            MBConv(24, 24, 3, 1, 6)
        )
        self.stage3 = nn.Sequential(
            MBConv(24, 40, 5, 2, 6),
            MBConv(40, 40, 5, 1, 6)
        )
        self.stage4 = nn.Sequential(
            MBConv(40, 80, 3, 2, 6),
            MBConv(80, 80, 3, 1, 6),
            MBConv(80, 80, 3, 1, 6)
        )
        self.stage5 = nn.Sequential(
            MBConv(80, 112, 5, 1, 6),
            MBConv(112, 112, 5, 1, 6),
            MBConv(112, 112, 5, 1, 6)
        )
        self.stage6 = nn.Sequential(
            MBConv(112, 192, 5, 2, 6),
            MBConv(192, 192, 5, 1, 6),
            MBConv(192, 192, 5, 1, 6),
            MBConv(192, 192, 5, 1, 6)
        )
        self.stage7 = MBConv(192, 320, 3, 1, 6)
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.head(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

# 创建模型实例并测试输入输出
model = EfficientNet(num_classes=10)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)
