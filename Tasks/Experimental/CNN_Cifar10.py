import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o
from Data import Cifar10Loader

class MyCNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(2*2*32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Input shape [3, 32, 32]
        """
        x = F.avg_pool2d(self.conv1(x), 2)  # [8, 14, 14]
        x = F.avg_pool2d(self.conv2(x), 2)  # [16, 6, 6]
        x = F.avg_pool2d(self.conv3(x), 2)  # [32, 2, 2]
        x = x.view(-1, 2*2*32)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


def mycnn_cifar10_experiment(hparams:dict=None, settings:dict=None):
    if hparams is None:
        hparams = dict()
    learning_rate = hparams.get("learning_rate", 0.01)
    batch_size = hparams.get("batch_size", 64)
    if settings is None:
        settings = dict()
    n_rounds = settings.get("n_rounds", 10000)
    dataloader = Cifar10Loader()
    my_cnn = MyCNN().cuda()
    loss_func = nn.CrossEntropyLoss()
