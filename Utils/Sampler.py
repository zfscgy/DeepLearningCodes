import numpy as np


class Sampler:
    def __init__(self, item_size, n_samples):
        self.item_size = item_size
        self.n_samples = n_samples

    def get_samples(self, batch_size):
        return [np.random.choice(self.item_size, self.n_samples, replace=False) for _ in range(batch_size)]