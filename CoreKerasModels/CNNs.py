import tensorflow as tf
L = tf.keras.layers
M = tf.keras.models
A = tf.keras.activations


class OmniglotCNN:
    """
    CNN for omniglot images
    Input size is [105, 105, 1]
    Notice:
        default stride for Conv layers = 1
        default padding for Conv is 'valid' (no padding)
    """
    def __init__(self):
        self.model = M.Sequential()
        self.model.add(L.Conv2D(filters=64, kernel_size=(10, 10), activation=A.relu))  # [96, 96, 64]
        self.model.add(L.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # [48, 48, 64]
        self.model.add(L.Conv2D(filters=128, kernel_size=(7, 7), activation=A.relu)) # [42, 42, 128]
        self.model.add(L.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # [21, 21, 128]
        self.model.add(L.Conv2D(filters=128, kernel_size=(4, 4), activation=A.relu))  # [18, 18, 128]
        self.model.add(L.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # [9, 9, 128]
        self.model.add(L.Conv2D(filters=256, kernel_size=(4, 4), activation=A.relu))  # [6, 6, 256]
        self.model.add(L.Flatten())  # [9216]
        self.model.add(L.Dense(4096, activation=A.sigmoid))  # [4096]

    def get_models(self):
        return [self.model]
