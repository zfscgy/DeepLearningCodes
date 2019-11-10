import tensorflow as tf
keras = tf.keras
L = keras.layers
K = keras.backend


class ReduceDimLayer(L.Layer):
    def __init__(self, keep_dim=1):
        self.keep_dim = keep_dim
        super(ReduceDimLayer, self).__init__()

    def build(self, input_shape):
        self.inputs_shape = input_shape
        super(ReduceDimLayer, self).build(input_shape)

    def call(self, inputs):
        return K.reshape(inputs, shape=[-1] + self.inputs_shape[-self.keep_dim:])
