import torch.nn as nn

from utils.builder import get_builder
from args import args

# structure described in http://proceedings.mlr.press/v97/kornblith19a/kornblith19a-supp.pdf
class CNN_10(nn.Module):
    def __init__(self, num_channels=3):
        super(CNN_10, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten()
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.dense(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)