import numpy as np
from PIL import Image
import os


class AnimeFacesLoader:
    def __init__(self, resize_shape):
        self.image_list = os.listdir("./Data/Datasets/AnimeFaces/Faces")
        self.image_num = len(self.image_list)
        self.resize_shape = resize_shape
        self.get_train_batch = self.get_batch

    def get_batch(self, batch_size):
        """

        """
        img_files = np.random.choice(self.image_list, batch_size)
        imgs = [Image.open(os.path.join("Data/Datasets/AnimeFaces/Faces", img)) for img in img_files]
        imgs = [img.resize(self.resize_shape) for img in imgs]
        imgs = [np.array(img).astype(np.float)/128 - 1 for img in imgs]
        return np.array(imgs)