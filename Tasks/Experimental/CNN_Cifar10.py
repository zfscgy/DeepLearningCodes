import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o
from Eval.Metrics import top1_accuracy
from Data import Cifar10Loader

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1_0 = nn.Conv2d(3, 8, 3)
        self.conv1_1 = nn.Conv2d(8, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(2*2*32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Input shape [3, 32, 32]
        """
        x = F.avg_pool2d(F.leaky_relu(self.conv1_1(self.conv1_0(x))), 2)  # [8, 14, 14]
        x = F.avg_pool2d(F.leaky_relu(self.conv2(x)), 2)  # [16, 6, 6]
        x = F.avg_pool2d(F.leaky_relu(self.conv3(x)), 2)  # [32, 2, 2]
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
    optimizer = o.Adam(my_cnn.parameters(), learning_rate)
    for i in range(n_rounds):
        if i % 100 == 0:
            test_xs, test_ys = dataloader.get_test_batch()
            test_xs = torch.from_numpy(test_xs).cuda()
            pred_test_ys = my_cnn(test_xs).cpu().detach().numpy()
            acc = top1_accuracy(test_ys, pred_test_ys)
            print("Batch {}, accuracy {:.4f}".format(i, acc))
        xs, ys = dataloader.get_train_batch(batch_size)
        xs = torch.from_numpy(xs).cuda()
        ys = torch.from_numpy(ys).long().cuda()
        pred_ys = my_cnn(xs)
        loss = loss_func(pred_ys, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    mycnn_cifar10_experiment(settings={"n_rounds": 20000})

