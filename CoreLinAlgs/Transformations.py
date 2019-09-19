import numpy as np


class PCA:
    """

    """
    def __init__(self, xs: np.ndarray):
        """

        :param xs: [n_samples, n_dim]
        """
        self.mul = np.matmul(xs.T, xs)
        self.sum = np.sum(xs, axis=0, keepdims=True)  # [1, n_dim]
        self.data_count = xs.shape[0]
        self.cov = None
        self.eig_vals = None
        self.eig_vecs = None

    def feed(self, xs: np.ndarray):
        assert self.sum.shape[0] == xs.shape[1], "Dimension of data should be identical"
        self.mul += np.sum(xs.T, xs)
        self.sum += np.sum(xs, axis=0, keepdims=True)
        self.data_count += xs.shape[0]

    def pca(self):
        self.cov = (self.mul - np.matmul(self.sum.T, self.sum)) / self.data_count
        # The eig_vecs is formed by column eigen vectors
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.cov)
        dec_order = np.argsort(-self.eig_vals)
        self.eig_vals = self.eig_vals[dec_order]
        self.eig_vecs = self.eig_vecs[:, dec_order]

    def get_portion(self, p:float):
        assert 0 < p < 1, "Portion must be between 0 and 1"
        portion = 0
        total_v = np.sum(self.eig_vals)
        for i in range(self.eig_vals.shape[0]):
            portion += self.eig_vals[i]
            if portion > p * total_v:
                return self.eig_vecs[:, :i + 1]
        return self.eig_vecs