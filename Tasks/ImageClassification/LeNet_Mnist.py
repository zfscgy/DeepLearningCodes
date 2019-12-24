import torch
import torch.nn as nn
import torch.optim as o
import numpy as np
from Data import MnistLoader
from Torch.Modules import LeNet5
from Eval.Metrics import top1_accuracy


def lenet_mnist_experiment(hparams:dict = None, settings:dict = None):
    if hparams is None:
        hparams = dict()
    learning_rate = hparams.get("learning_rate", 0.01)
    batch_size = hparams.get("batch_size", 64)
    if settings is None:
        settings = dict()
    n_rounds = settings.get("n_rounds", 10000)

    dataloader = MnistLoader()
    lenet = LeNet5().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = o.Adam(lenet.parameters(), lr=learning_rate)
    for i in range(n_rounds):
        if i % 100 == 0:
            test_xs, test_ys = dataloader.get_test_batch()
            test_xs = test_xs.reshape([-1, 1, 28, 28])
            test_xs = torch.from_numpy(test_xs).cuda()
            pred_test_ys = lenet(test_xs).cpu().detach().numpy()
            accuracy = top1_accuracy(test_ys, pred_test_ys)
            print("Rounds:{}, test accuracy {:.4f}".format(i, accuracy))
        xs, ys = dataloader.get_train_batch(batch_size)
        xs = xs.reshape([-1, 1, 28, 28])  # [batch, 1, 28, 28]
        xs = torch.from_numpy(xs).cuda()
        ys = torch.from_numpy(ys).long().cuda()
        pred_ys = lenet(xs)
        loss = loss_func(pred_ys, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    lenet_mnist_experiment()
