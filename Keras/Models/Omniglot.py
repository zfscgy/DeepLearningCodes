import tensorflow as tf
k = tf.keras
L = tf.keras.layers
M = tf.keras.models
A = tf.keras.activations
Lo = tf.keras.losses
K = tf.keras.backend

class OmniglotCNN(k.Model):
    """
    CNN for omniglot images
    Input size is [105, 105, 1]
    Output size is [4096]
    Notice:
        default stride for Conv layers = 1
        default padding for Conv is 'valid' (no padding)
    """
    def __init__(self):
        super(OmniglotCNN, self).__init__()
        self.conv_1 = L.Conv2D(filters=16, kernel_size=(10, 10), activation=A.relu, input_shape=(105, 105, 1))  # [96, 96, 16]
        self.maxpool_1 = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2))  # [48, 48, 16]
        self.conv_2 = L.Conv2D(filters=32, kernel_size=(7, 7), activation=A.relu) # [42, 42, 32]
        self.maxpool_2 = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2))  # [21, 21, 128]
        self.conv_3 = L.Conv2D(filters=64, kernel_size=(4, 4), activation=A.relu)  # [18, 18, 64]
        self.maxpool_3 = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2))  # [9, 9, 128]
        self.conv_4 = L.Conv2D(filters=128, kernel_size=(4, 4), activation=A.relu)  # [6, 6, 128]
        self.flat = L.Flatten()  # [9216]
        self.dense = L.Dense(64, activation=A.sigmoid)  # [64]

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.maxpool_1(x)
        x = self.maxpool_2(self.conv_2(x))
        x = self.maxpool_3(self.conv_3(x))
        x = self.conv_4(x)
        x = self.dense(self.flat(x))
        return x


class OmniglotSiameseNetwork(k.Model):
    """
    Input shape: [(105, 105, 1), (105, 105, 1)]
    Output shape: (1,)
    """
    def __init__(self):
        super(OmniglotSiameseNetwork, self).__init__()
        self.cnn = OmniglotCNN()
        self.feature_diff = L.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        self.dense_out = L.Dense(1, activation=A.sigmoid, use_bias=False)
        self.build([(None, 105, 105, 1), (None, 105, 105, 1)])

    def call(self, inputs, training=None, mask=None):
        i0, i1 = inputs
        x0 = self.cnn(i0)
        x1 = self.cnn(i1)
        d = self.feature_diff([x0, x1])
        return self.dense_out(d)
