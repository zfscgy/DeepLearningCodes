import numpy as np
from PIL import Image
import pathlib


class LFWLoader:
    def __init__(self, split=0.8):
        path = "./Data/Datasets/LFW/lfw-deepfunneled"
        self.paths = list(pathlib.Path(path).iterdir())
        self.train_len = int(len(self.paths) * split)

    def _get_batch(self, batch_size, support_size=2, train=True):
        """
        Return shape:
        [batch, 250, 250, 3]
        [batch, 250, 250, 3]
        [batch]
        """
        if train:
            paths = self.paths[:self.train_len]
        else:
            paths = self.paths[self.train_len:]
            if batch_size is None:
                batch_size = len(paths)
        folders = np.random.choice(paths, support_size, replace=False).tolist()
        imgs0 = []
        imgs1 = []
        labels = []
        for _ in range(batch_size):
            same_img = np.random.choice(list(folders[0].iterdir()), 2, replace=False)
            same_img[0] = np.asarray(Image.open(same_img[0]))
            same_img[1] = np.asarray(Image.open(same_img[1]))
            imgs0.append(same_img[0])
            imgs1.append(same_img[1])
            labels.append(1)
            for folder in folders[1:]:
                img0 = same_img[0]
                img1path = np.random.choice(list(folder.iterdir()))
                img1 = np.asarray(Image.open(img1path))
                imgs0.append(img0)
                imgs1.append(img1)
                labels.append(0)
        return (np.array(imgs0).astype(np.float32) - 128)/255, \
               (np.array(imgs1).astype(np.float32) - 128)/255, \
               np.array(labels)

    def get_train_batch(self, batch_size, support_size=2):
        return self._get_batch(batch_size, support_size, train=True)

    def get_test_batch(self, batch_size=None, support_size=2):
        return self._get_batch(batch_size, support_size, train=False)

