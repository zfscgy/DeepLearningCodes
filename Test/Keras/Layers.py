import tensorflow as tf
L = tf.keras.layers
A = tf.keras.activations
K = tf.keras.backend
Ops = tf.keras.optimizers
Losses = tf.keras.losses
import numpy as np

class MyDenseLayer(L.Layer):
    def __init__(self, units):
        self.units = units
        self.dense_layer = L.Dense(self.units)
        super(MyDenseLayer, self).__init__()


    def build(self, input_shape):
        self.dense_layer.build(input_shape)
        # Must add following two lines, otherwise the nested dense layer won't be trained
        self._trainable_weights += self.dense_layer.trainable_weights
        self._non_trainable_weights += self.dense_layer.non_trainable_weights
        super(MyDenseLayer, self).build(input_shape)
        print(self._trainable_weights)

    def call(self, inputs, **kwargs):
        return self.dense_layer.call(inputs)

    def compute_output_shape(self, input_shape):
        return list(input_shape[:-1]).append(self.units)


x1s = np.random.uniform(-1, 1, [1000, 1])
x2s = np.random.uniform(0, 1, [1000, 1])
zs = x1s + x2s
xs = np.hstack([x1s, x2s])


model = tf.keras.Sequential()
model.add(MyDenseLayer(1))
model.compile(Ops.SGD(0.1), Losses.mean_squared_error)
model.fit(xs, zs, batch_size=30, epochs=1000)