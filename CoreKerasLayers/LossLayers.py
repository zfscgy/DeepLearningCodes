import tensorflow as tf
keras = tf.keras
L = keras.layers
A = keras.activations
K = keras.backend
I = keras.initializers
import numpy as np

class SampleOutputLayer(L.Layer):
    """
    The inputs of this layer is
    1. label [batch, 1]
    2. sample_indices [batch, n_sampls]
    3. output_probalities [batch, item_size]
    """
    def __init__(self, sample_num, leading_dims=0):
        self.sample_num = sample_num
        self.leading_dims = leading_dims
        super(SampleOutputLayer, self).__init__()
        
    def build(self, input_shape):
        y_true = np.array([1] + [0] * self.sample_num)
        # always a [1, 0, 0, 0, ..., 0] tensor
        self.targets = self.add_weight(name="target", shape=(1 + self.sample_num,),
                                 initializer=I.Constant(value=y_true), trainable=False)
        super(SampleOutputLayer, self).build(input_shape)

    def call(self, inputs):
        labels, samples, probs = inputs
        indices = K.concatenate([labels, samples], axis=-1)  # [batch, 1 + n_samples]
        indices = K.expand_dims(indices, -1)  # [batch, 1 + n_samples, 1]
        # [batch.., 1 + n_samples, 1], we have to expand because the last dim specifies index dim in tf.gather_nd
        gathered_probs = tf.gather_nd(probs, indices, batch_dims=1 + self.leading_dims)
        # [batch.., 1 + n_samples]
        targets = self.targets + K.zeros_like(gathered_probs)
        # [batch, 1 + samples]
        return targets, gathered_probs