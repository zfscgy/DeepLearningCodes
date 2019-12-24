import torch
import torch.optim as o
from Torch.Modules import LFWSiameseNetwork
from Data import LFWLoader


def lfw_experiment(hparams:dict=None, settings:dict=None, verbose=1):
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

    lfw_net = LFWSiameseNetwork()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = o.Adam(lfw_net.parameters(), lr=learning_rate)

    for i in range(n_rounds):
        xs, ys = dataloader.get_train_batch(batch_size, support_size)
        pred_ys = lfw_net(xs)
        loss = loss_func(pred_ys, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n_rounds % 100 == 0:
            test_xs, test_ys = dataloader.get_test_batch(test_batch_size, support_size)
            pred_test_ys = lfw_net(test_xs)
