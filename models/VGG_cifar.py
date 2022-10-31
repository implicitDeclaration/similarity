import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VGG', 'cvgg11_bn', 'cvgg13_bn', 'cvgg16_bn', 'cvgg19_bn']


class ConvStitch(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ConvStitch, self).__init__()
        self.stitch_layer = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1)

    def forward(self, x):
        y = self.stitch_layer(x)
        y = F.relu(y)
        return y


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x, mid_input=False, mid_output=False):
        if not mid_input:
            if mid_output:
                assert not (mid_input and mid_output)
                for i in range(mid_output + 1):
                    x = self.features[i](x)
                return x
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out
        else:
            for i in range(len(self.features)):
                if i < mid_input:
                    continue
                else:
                    x = self.features[i](x)
            x = x.view(x.size(0), -1)
            out = self.classifier(x)
            return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
#
# class VGG(nn.Module):
#     '''
#     VGG model
#     '''
#     def __init__(self, features, num_classes=10):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, num_classes),
#         )
#          # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 m.bias.data.zero_()
#
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def cvgg11_bn(num_classes, batch_norm=False):
    return VGG(make_layers(cfg['A'], batch_norm=batch_norm), num_classes=num_classes)


def cvgg13_bn(num_classes, batch_norm=False):
    return VGG(make_layers(cfg['B'], batch_norm=batch_norm), num_classes=num_classes)


def cvgg16_bn(num_classes, batch_norm=False):
    return VGG(make_layers(cfg['D'], batch_norm=batch_norm), num_classes=num_classes)


def cvgg19_bn(num_classes, batch_norm=False):
    return VGG(make_layers(cfg['E'], batch_norm=batch_norm), num_classes=num_classes)


if __name__ == "__main__":
    import torch

    inter_feature = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    input = torch.randn(2, 3, 32, 32)
    model = cvgg19_bn(10, batch_norm=True)
    print(model)
    output = model(input)
    print(output)


