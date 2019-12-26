import numpy as np
import torch
import torch.optim as o
from Torch.Modules import LFWSiameseNetwork
from Data import LFWLoader
from Eval.Metrics import auc, oneshot_accuracy

def lfw_experiment(hparams:dict=None, settings:dict=None):
    if hparams is None:
        hparams = dict()
    batch_size = hparams.get("batch_size", 32)
    learning_rate = hparams.get("learning_rate", 0.01)
    if settings is None:
        settings = dict()
    support_size = settings.get("support_size", 2)
    n_rounds = settings.get("n_rounds", 2000)
    train_test_split = settings.get("train_test_split", 0.8)
    test_batch_size = settings.get("test_batch_size", 1000)
    dataloader = LFWLoader(split=train_test_split)

    lfw_net = LFWSiameseNetwork().cuda()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = o.Adam(lfw_net.parameters(), lr=learning_rate)

    for i in range(n_rounds):
        if i % 100 == 0:
            x0s, x1s, ys = dataloader.get_test_batch(test_batch_size, support_size)
            x0s = np.swapaxes(x0s, 1, 3).swapaxes(2, 3)
            x0s = torch.from_numpy(x0s).cuda()
            x1s = np.swapaxes(x1s, 1, 3).swapaxes(2, 3)
            x1s = torch.from_numpy(x1s).cuda()
            pred_ys = lfw_net([x0s, x1s]).cpu().detach().numpy()
            acc = oneshot_accuracy(pred_ys, support_size)
            print("Round {} Accuracy {:.4f}".format(i, acc))
        x0s, x1s, ys = dataloader.get_train_batch(batch_size, support_size)
        x0s = np.swapaxes(x0s, 1, 3).swapaxes(2, 3)
        x0s = torch.from_numpy(x0s).cuda()
        x1s = np.swapaxes(x1s, 1, 3).swapaxes(2, 3)
        x1s = torch.from_numpy(x1s).cuda()
        ys = torch.from_numpy(ys).long().cuda()
        pred_ys = lfw_net([x0s, x1s])
        loss = loss_func(pred_ys, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    lfw_experiment(settings={"test_batch_size": 32})
