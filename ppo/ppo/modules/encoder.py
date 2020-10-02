from torch import nn
import torch

class Encoder(nn.Module):

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, x):

        out = x

        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = self.conv3(out)
        out = nn.functional.relu(out)
        out = self.conv4(out)
        out = nn.functional.relu(out)

        out = out.reshape(-1, 1024)

        return out