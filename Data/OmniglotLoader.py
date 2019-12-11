# from https://github.com/Goldesel23/Siamese-Networks-for-One-Shot-Learning
import os
import random
import numpy as np
import math
from PIL import Image
import pathlib
from Data.Tools import ImageAugmentor


class OmniglotLoader:
    def __init__(self):
        self.base_path = pathlib.Path("./Data/Datasets/Omniglot/Omniglot Dataset")
        self.train_paths = list(self.base_path.joinpath("images_background").iterdir())
        self.test_paths = list(self.base_path.joinpath("images_evaluation").iterdir())
        rotation_range = [-15, 15]
        shear_range = [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi]
        zoom_range = [0.8, 2]
        shift_range = [5, 5]
        self.image_augumentor = ImageAugmentor(0.5, shear_range, rotation_range, shift_range, zoom_range)

    def _get_batch_from_folder(self, folder, batch_size, support_size):
        """
        return shape:
        [batch, 105, 105, 1],
        [batch, 105, 105, 1],
        [batch, 1]
        """
        alphebets = np.random.choice(folder, batch_size)
        image_pairs = [[], []]
        labels = []
        for alphabet in alphebets:
            # Choose random characters
            chs = np.random.choice(list(alphabet.iterdir()), support_size, replace=False).tolist()
            chs.append(chs[0])
            # From character folder, choose random images
            chs = [np.random.choice(list(ch.iterdir())) for ch in chs]
            for i in range(support_size):
                """
                Attention! The omniglot images only has two values: 0 and 1, using np.asarray will generate bool arrays
                The value scope is [0, 1], not [0, 255]
                """
                image_pairs[0].append((np.asarray(Image.open(chs[i])) - 0.5) / 0.5)
                image_pairs[1].append((np.asarray(Image.open(chs[support_size])) - 0.5) / 0.5)
            label = [1] + [0] * (support_size - 1)
            labels.extend(label)
        return image_pairs, labels

    def get_train_batch(self, batch_size, support_size=2, use_augumentor=False):
        image_pairs, labels = self._get_batch_from_folder(self.train_paths, batch_size, support_size)
        if use_augumentor:
            self.image_augumentor.get_random_transform(image_pairs[0])
            self.image_augumentor.get_random_transform(image_pairs[1])
        return [np.asarray(image_pairs[0])[:, :, :, np.newaxis], np.asarray(image_pairs[1])[:, :, :, np.newaxis]], \
               np.asarray(labels, dtype=np.float)[:, np.newaxis]

    def get_test_batch(self, batch_size=None, support_size=2):
        # Choose random alphabets
        if batch_size is None:
            batch_size = len(self.test_paths)
        image_pairs, labels = self._get_batch_from_folder(self.train_paths, batch_size, support_size)
        return [np.asarray(image_pairs[0])[:, :, :, np.newaxis], np.asarray(image_pairs[1])[:, :, :, np.newaxis]], \
               np.asarray(labels, dtype=np.float)[:, np.newaxis]
