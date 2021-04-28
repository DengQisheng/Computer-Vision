import torch.nn as nn

# Basic block
class BasicBlock(nn.Module):

    factor = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        proc_channels = out_channels // BasicBlock.factor

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, proc_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(proc_channels),
            nn.ReLU(True),
            nn.Conv2d(proc_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Sequential()
        self.activation = nn.ReLU(True)

    def forward(self, x):
        identity = self.identity(x)
        residual = self.residual(x)
        return self.activation(residual + identity)


# Bottleneck block
class BottleneckBlock(nn.Module):

    factor = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()

        proc_channels = out_channels // BottleneckBlock.factor

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, proc_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(proc_channels),
            nn.ReLU(True),
            nn.Conv2d(proc_channels, proc_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(proc_channels),
            nn.ReLU(True),
            nn.Conv2d(proc_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Sequential()
        self.activation = nn.ReLU(True)

    def forward(self, x):
        identity = self.identity(x)
        residual = self.residual(x)
        return self.activation(residual + identity)


# ResNet
class ResNet(nn.Module):

    def __init__(self, block, layer, num_channels=1, num_classes=10):
        super(ResNet, self).__init__()

        self.input = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv = nn.Sequential(
            # Layer 1
            block(64, 64 * block.factor),
            *[block(64 * block.factor, 64 * block.factor) for _ in range(1, layer[0])],
            # Layer 2
            block(64 * block.factor, 128 * block.factor, stride=2),
            *[block(128 * block.factor, 128 * block.factor) for _ in range(1, layer[1])],
            # Layer 3
            block(128 * block.factor, 256 * block.factor, stride=2),
            *[block(256 * block.factor, 256 * block.factor) for _ in range(1, layer[2])],
            # Layer 4
            block(256 * block.factor, 512 * block.factor, stride=2),
            *[block(512 * block.factor, 512 * block.factor) for _ in range(1, layer[3])]
        )
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * block.factor, num_classes)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ResNet 18/34/50/101/152
def resnet(depth, **kwargs):

    print(f'Network: ResNet{depth}')
    print('-' * 80)

    if depth == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2])
    elif depth == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3])
    elif depth == 50:
        return ResNet(BottleneckBlock, [3, 4, 6, 3])
    elif depth == 101:
        return ResNet(BottleneckBlock, [3, 4, 23, 3])
    elif depth == 152:
        return ResNet(BottleneckBlock, [3, 8, 36, 3])
    else:
        return ResNet(**kwargs)