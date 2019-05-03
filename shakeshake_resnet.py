import torch.nn as nn
import torch.nn.functional as F

from shake_shake_function import get_alpha_beta, shake_function


class ResidualPath(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualPath, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        return self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.residual_path1 = ResidualPath(in_planes, planes, stride)
        self.residual_path2 = ResidualPath(in_planes, planes, stride)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        x1 = self.residual_path1(x)
        x2 = self.residual_path2(x)

        if self.training:
            shake_config = (True, True, True)
        else:
            shake_config = (False, False, False)

        alpha, beta = get_alpha_beta(x.size(0), shake_config, x.device)
        y = shake_function(x1, x2, alpha, beta)
        out = F.relu(
            y + self.shortcut(x)
        )
        return out

class SSResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SSResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._init_sub_block(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._init_sub_block(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._init_sub_block(block, 64, num_blocks[2], stride=2)
        self.layers = [self.layer1, self.layer2, self.layer3]

        self.linear = nn.Linear(64, num_classes)

    def _init_sub_block(self, block, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def SSResNet20():
    return SSResNet(BasicBlock, [3, 3, 3])
