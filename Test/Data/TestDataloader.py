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

test_omniglot_loader()
