import pickle
import numpy as np


class Cifar10Loader:
    def __init__(self):
        root = "./Data/Datasets/Cifar10/"
        train_sets = [pickle.load(open(root + "data_batch_" + str(i), "rb"), encoding="bytes") for i in range(1, 6)]
        train_data = np.vstack([train_set[b"data"] for train_set in train_sets])
        train_label = np.vstack([np.array(train_set[b"labels"])[:, np.newaxis] for train_set in train_sets])
        self.train_set = np.hstack([train_data, train_label])
        test_set = pickle.load(open(root + "test_batch", "rb"), encoding="bytes")
        test_data = test_set[b"data"]
        test_label = np.array(test_set[b"labels"])[:, np.newaxis]
        self.test_set = np.hstack([test_data, test_label])

    def get_train_batch(self, batch_size):
        """
        return shape: [batch, 3, 32, 32], [batch]
        """
        idx = np.random.choice(self.train_set.shape[0], batch_size)
        train_batch = self.train_set[idx]
        return (train_batch[:, :3072].reshape([-1, 3, 32, 32]).astype(np.float32) - 128)/255, train_batch[:, 3072]

    def get_test_batch(self, batch_size=None):
        if batch_size is None:
            return (self.test_set[:, :3072].reshape([-1, 3, 32, 32]).astype(np.float32) - 128)/255, \
                   self.test_set[:, 3072]
        idx = np.random.choice(self.test_set.shape[0], batch_size)
        test_batch = self.test_set[idx]
        return (test_batch[:, :3072].reshape([-1, 3, 32, 32]).astype(np.float32) - 128)/255, test_batch[:, 3072]
