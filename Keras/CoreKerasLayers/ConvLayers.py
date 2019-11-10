import tensorflow as tf
keras = tf.keras
L = keras.layers
A = keras.activations


class NextRecDilated1DResBlock(L.Layer):
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
        self.channels = channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.dilated_conv_1 = L.Conv1D(filters=channels, kernel_size=kernel_size,
                                       dilation_rate=dilation, padding='causal')
        self.layer_norm_1 = L.LayerNormalization()
        self.layer_relu_1 = L.LeakyReLU()

        self.dilated_conv_2 = L.Conv1D(filters=channels, kernel_size=kernel_size,
                                       dilation_rate=2 * dilation, padding='causal')
        self.layer_norm_2 = L.LayerNormalization()
        self.layer_relu_2 = L.LeakyReLU()
        super(NextRecDilated1DResBlock, self).__init__()

    def call(self, inputs):
        # input_shape is [batch, t, k], t is sequence len, k is feature dimension
        conv_1_out = self.dilated_conv_1(inputs)  # [batch, t, k]
        normed_1_out = self.layer_norm_1(conv_1_out)  # [batch, t, k]
        relu_1_out = self.layer_relu_1(normed_1_out)
        conv_2_out = self.dilated_conv_2(relu_1_out)  # [batch, t, k]
        normed_2_out = self.layer_norm_2(conv_2_out)  # [batch, t, k]
        relu_2_out = self.layer_relu_2(normed_2_out)
        return relu_2_out + inputs  # residual connection
