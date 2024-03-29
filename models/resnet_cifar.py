"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder
from args import args


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(in_planes, planes, stride=stride)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=1)
        self.bn2 = builder.batchnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(in_planes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, self.expansion * planes)
        self.bn3 = builder.batchnorm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, builder, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.builder = builder

        self.conv1 = builder.conv3x3(3, 64, stride=1, first_layer=True)
        self.bn1 = builder.batchnorm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = builder.conv1x1(512 * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, mid_input=False, mid_output=False):
        assert not (mid_input and mid_output)
        if not mid_input:
            out0 = F.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out0)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out = F.avg_pool2d(out4, 4)
            out = self.fc(out)
            if mid_output == 1:
                return out0
            elif mid_output == 2:
                return out1
            elif mid_output == 3:
                return out2
            elif mid_output == 4:
                return out3
            return out.flatten(1)
        else:
            if mid_input == 1:
                out = self.layer1(x)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
            elif mid_input == 2:
                out = self.layer2(x)
                out = self.layer3(out)
                out = self.layer4(out)
            elif mid_input == 3:
                out = self.layer3(x)
                out = self.layer4(out)
            else:
                out = self.layer4(x)
            out = F.avg_pool2d(out, 4)
            out = self.fc(out)
            return out.flatten(1)


def cResNet18(num_classes):
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2])

def cResNet34(num_classes):
    return ResNet(get_builder(), BasicBlock, [3, 4, 6, 3])


def cResNet50(num_classes):
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3])


def cResNet101(num_classes):
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3])


def cResNet152(num_classes):
    return ResNet(get_builder(), Bottleneck, [3, 8, 36, 3])

