import tensorflow as tf
import numpy as np
from Keras.Components.LossLayers import SampledCrossEntropy, MaxBPRLoss
k = tf.keras
L = tf.keras.layers
M = tf.keras.models
A = tf.keras.activations
Lo = tf.keras.losses
K = tf.keras.backend


class NextItemResBlock(L.Layer):
    '''
    This layer is described in "A Simple Convolutional Network for Next Item Recommendation"
    '''
    def __init__(self, channels, dilation, kernel_size):
        """

        :param channels:
        :param dilation:
        :param kernel_size:

        The dilation for first convolution layer is 'dilation'
        And the dilation for second convolution layer is '2 * dilation'
        The input is a sequence with shape [batch, t, k]    t: sequence length, k:feature_size
        The output is the same shape and shifted one time step
        for example: input represents [3,4,5,6,7]
        the output is the prediction of [4,5,6,7,8]
        and the prediction at time t will not use input at time t, t+1 .... (causal convolution)
        """
        super(NextItemResBlock, self).__init__()
        self.channels = channels
        self.dilation = dilation
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.dilated_conv_1 = L.Conv1D(filters=self.channels, kernel_size=self.kernel_size,
                                       dilation_rate=self.dilation, padding='causal', input_shape=input_shape)
        self.layer_norm_1 = L.LayerNormalization()
        self.layer_relu_1 = L.LeakyReLU()

        self.dilated_conv_2 = L.Conv1D(filters=self.channels, kernel_size=self.kernel_size,
                                       dilation_rate=2 * self.dilation, padding='causal')
        self.layer_norm_2 = L.LayerNormalization()
        self.layer_relu_2 = L.LeakyReLU()

    def call(self, inputs):
        # input_shape is [batch, t, k], t is sequence len, k is feature dimension
        x = self.dilated_conv_1(inputs)  # [batch, t, k]
        x = self.layer_norm_1(x)  # [batch, t, k]
        x = self.layer_relu_1(x)
        x = self.dilated_conv_2(x)  # [batch, t, k]
        x = self.layer_norm_2(x)  # [batch, t, k]
        x = self.layer_relu_2(x)
        return x + inputs  # residual connection


class NextItemCNN(k.Model):
    def __init__(self, item_size, item_feature_dim, sequence_length, res_blocks):
        """
        :param item_feature_dim:
        :param item_size:
        :param sequence_length:
        :param res_blocks: a list represents every resblock's first conv layer's dilation and kernel size
        (second layer is 2 times dilation). According to the author, it's best [(1, 2), (2, 2)]
        """
        super(NextItemCNN, self).__init__()
        self.item_embeddings = L.Embedding(item_size, item_feature_dim, input_length=sequence_length)
        self.blocks = []
        for dilation, kernel_size in res_blocks:
            res_block_layer = NextItemResBlock(item_feature_dim, dilation, kernel_size)
            self.blocks.append(res_block_layer)
        self.dense = L.Dense(item_size)

    def call(self, inputs, training=None, mask=None):
        x = self.item_embeddings(inputs)
        for res_block in self.blocks:
            x = res_block(x)
        x = self.dense(x)
        return x


class NextItCNN_CE(k.Model):
    def __init__(self, item_size, item_feature_dim, sequence_length, res_blocks, n_negative_samples, batch_size):
        super(NextItCNN_CE, self).__init__()
        self.pred_model = NextItemCNN(item_size, item_feature_dim, sequence_length, res_blocks)
        self.sampled_cross_entropy = SampledCrossEntropy(n_negative_samples, batch_size)
        self.predict([np.zeros([batch_size, sequence_length], dtype=np.int),
                      np.zeros([batch_size, sequence_length], dtype=np.int)])

    def call(self, inputs, training=None, mask=None):
        xs, ys = inputs[0], inputs[1]
        pred_logits = self.pred_model(xs)
        loss = self.sampled_cross_entropy([pred_logits, ys])
        return loss