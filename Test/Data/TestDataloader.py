import numpy as np
import matplotlib.pyplot as plt
from Data import *
def test_omniglot_loader():
    loader = OmniglotLoader()
    [x1s, x2s], ys = loader.get_train_batch(2)
    plt.imshow(x1s[0, :, :, 0])
    plt.show()
    plt.imshow(x2s[0, :, :, 0])
    plt.show()
    plt.imshow(x1s[1, :, :, 0])
    plt.show()
    plt.imshow(x2s[1, :, :, 0])
    plt.show()
    print(ys)

def test_mnist_loader():
    loader = MnistLoader()
    print(loader.train_set.shape)
    print(loader.test_set.shape)
    x, y = loader.get_train_batch(1)
    x = x[0, :]
    y = y[0, 0]
    plt.imshow(x.reshape([28, 28]))
    plt.show()
    print(y)
    x, y = loader.get_test_batch(1)
    x = x[0, :]
    y = y[0, 0]
    plt.imshow(x.reshape([28, 28]))
    plt.show()
    print(y)

def test_cifar10_loader():
    loader = Cifar10Loader()
    x, y = loader.get_train_batch(1)
    x = x[0]
    y = y[0]
    x = np.swapaxes(x, 0, 2).swapaxes(1, 0)
    plt.imshow(x)
    plt.show()
    print(y)
    x, y = loader.get_test_batch(1)
    x = x[0]
    y = y[0]
    x = np.swapaxes(x, 0, 2).swapaxes(1, 0)
    plt.imshow(x)
    plt.show()
    print(y)

def test_lfw_loader():
    loader = LFWLoader()
    x0s, x1s, ys = loader.get_train_batch(2)
    x0 = x0s[0]
    x1 = x1s[0]
    y = ys[0]
    print(y)
    plt.imshow(x0)
    plt.show()
    plt.imshow(x1)
    plt.show()
    x0 = x0s[1]
    x1 = x1s[1]
    y = ys[1]
    print(y)
    plt.imshow(x0)
    plt.show()
    plt.imshow(x1)
    plt.show()
    x0, x1, y = loader.get_test_batch(1)
    x0 = x0[0]
    x1 = x1[0]
    y = y[0]
    plt.imshow(x0)
    plt.show()
    plt.imshow(x1)
    plt.show()
    print(y)

if __name__ == "__main__":
    test_lfw_loader()