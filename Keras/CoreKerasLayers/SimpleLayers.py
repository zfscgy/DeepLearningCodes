import tensorflow as tf
keras = tf.keras
L = keras.layers
A = keras.activations
K = keras.backend


class SoftMaxWithEmbedding(L.Layer):
    def __init__(self, item_size, embeddings_layer, bias_initializer=None, bias_regularizer=None, bias_constraint=None):
        self.softmax = L.Softmax()
        self.item_size = item_size
        self.embeddings = embeddings_layer
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        super(SoftMaxWithEmbedding, self).__init__()

    def build(self, input_shape):
        self.bias = self.add_weight("softmax_bias", (self.item_size,),
                                    initializer=self.bias_initializer, regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        self.embeddings = K.transpose(self.embeddings.weights[0])
        super(SoftMaxWithEmbedding, self).build(input_shape)

    def call(self, inputs):
        # input [batch, sequence_length, feature_dim]
        probs = K.dot(inputs, self.embeddings)
        probs = probs + self.bias
        return probs