import tensorflow as tf
L = tf.keras.layers
M = tf.keras.models
A = tf.keras.activations
Lo = tf.keras.losses
K = tf.keras.backend

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


from CoreKerasLayers.ConvLayers import NextRecDilated1DResBlock as _resBlock
from CoreKerasLayers.SimpleLayers import SoftMaxWithEmbedding as _softmaxEmb
class NextItemCNN:
    def __init__(self, item_feature_dim, item_size, sequence_length, res_blocks):
        """

        :param item_feature_dim:
        :param item_size:
        :param sequence_length:
        :param res_blocks: a list represents every resblock's first conv layer's dilation
        (second layer is 2 times dilation). According to the author, it's best [1, 2]
        """
        self.inputs = L.Input(shape=(sequence_length,))  # [batch_size, seq_len]
        self.item_embeddings = L.Embedding(item_size, item_feature_dim, input_length=sequence_length)
        self.embedding_seq = self.item_embeddings(self.inputs)  # [batch_size, seq_len, feature_dim]
        self.res_block_layers = []
        feature_seq = self.embedding_seq
        for dilation, kernel_size in res_blocks:
            res_block_layer = _resBlock(item_feature_dim, dilation, kernel_size)
            feature_seq = res_block_layer(feature_seq)  # [batch_size, seq_len, feature_dim]

        # self.final_softmax_layer = _softmaxEmb(item_size, self.item_embeddings)
        self.output_logits_layer = L.Dense(item_size)
        self.output_logits = self.output_logits_layer(feature_seq)

        self.output_softmax_layer = L.Softmax()
        self.output_probs = self.output_softmax_layer(self.output_logits)
        # [batch_size, seq_len, item_size]
        self.model = M.Model(inputs=self.inputs, outputs=self.output_probs)
        self.model_logits = M.Model(inputs=self.inputs, outputs=self.output_logits)