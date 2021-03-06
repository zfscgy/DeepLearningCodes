import torch
import torch.nn as nn
import torch.nn.functional as F


class LFWCNN(nn.Module):
    """
    Input size: [batch, 3, 250, 250]
    Ouput size: [batch, 128]
    """
    def __init__(self):
        super(LFWCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # [batch, 8, 250, 250]
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)  # [batch, 8, 125, 125]
        self.conv2_0 = nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=0)  # [batch, 12, 123, 123]
        self.conv2_1 = nn.Conv2d(12, 16, kernel_size=4, stride=1, padding=0)  # [batch, 16, 120, 120]
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)  # [batch, 16, 60, 60]
        self.conv3 = nn.Conv2d(16, 20, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)  # [batch, 16, 30, 30]
        self.conv4 = nn.Conv2d(20, 24, kernel_size=3, stride=1, padding=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2)  # [batch, 20, 15, 15]
        self.conv5_0 = nn.Conv2d(24, 32, kernel_size=3, stride=1)
        self.conv5_1 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
        self.max_pool5 = nn.MaxPool2d(kernel_size=2)  # [batch, 24, 5, 5]
        self.dense = nn.Linear(640, 128)

    def forward(self, x: torch.Tensor):
        x = self.max_pool1(F.relu(self.conv1(x)))
        x = self.max_pool2(F.relu(self.conv2_1(self.conv2_0(x))))
        x = self.max_pool3(F.relu(self.conv3(x)))
        x = self.max_pool4(F.relu(self.conv4(x)))
        x = self.max_pool5(F.relu(self.conv5_1(self.conv5_0(x))))
        x = x.view(-1, 640)
        x = F.sigmoid(self.dense(x))
        return x


class LFWSiameseNetwork(nn.Module):
    def __init__(self):
        super(LFWSiameseNetwork, self).__init__()
        self.cnn = LFWCNN()
        self.dense = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        x0, x1 = x
        f0 = self.cnn(x0)
        f1 = self.cnn(x1)
        diff = torch.abs(f0 - f1)
        return F.sigmoid(self.dense(diff))