import tensorflow as tf
keras = tf.keras
L = keras.layers
Lo = keras.losses
A = keras.activations
K = keras.backend
from Keras.CoreKerasLayers.LossLayers import SampleOutputLayer as _sampler

class SampledCrossEntropy:
    '''
    Considered extra dimensions
    '''
    def __init__(self, n_negative_samples, n_item_size, leading_shape=None):
        if leading_shape is None:
            leading_shape = []

        self.labels = L.Input(leading_shape, dtype='int32')
        self.sample_indices = L.Input(leading_shape + [n_negative_samples], dtype='int32')
        self.output_ratings = L.Input(leading_shape + [n_item_size])
        self.labels_expanded = K.expand_dims(self.labels, -1)
        self.sampler = _sampler(n_negative_samples, len(leading_shape))
        self.sampled_targets, self.sampled_ratings = \
           self.sampler([self.labels_expanded, self.sample_indices, self.output_ratings])
        # [batch, leading_shape.., item_size]
        self.sampled_probs = K.softmax(self.sampled_ratings)
        self.sampled_crossentropy_loss = Lo.categorical_crossentropy(self.sampled_targets, self.sampled_probs)
        # [batch, leading_shape, 1]
        self.sampled_crossentropy_loss = K.expand_dims(K.mean(self.sampled_crossentropy_loss), axis=0)
        self.model = keras.Model([self.labels, self.sample_indices, self.output_ratings], self.sampled_crossentropy_loss)


class MaxBPRLoss:
    """
    This is the best loss function according to the paper
    'Recurrent Neural Networks for Top-k Recommendation'
    Note this class uses tf.gather_nd
    """
    def __init__(self, n_negative_samples, n_item_size, leading_shape=None):
        if leading_shape is None:
            leading_shape = []
        self.labels = L.Input(leading_shape, dtype='int32')
        self.labels_expanded = K.expand_dims(self.labels, -1)
        self.sample_indices = L.Input(leading_shape + [n_negative_samples], dtype='int32')
        self.output_ratings = L.Input(leading_shape + [n_item_size])
        self.sampler = _sampler(n_negative_samples, len(leading_shape))
        self.sampled_targets, self.sampled_ratings = \
           self.sampler([self.labels_expanded, self.sample_indices, self.output_ratings])
        def max_bpr_loss(ys):
            ys_true, ys_pred = ys   # [batch.., 1 + n_samples]
            probs_ys_pred = K.softmax(ys_pred)
            #                                             [batch.., n_samples]          [batch, ..]
            sigmoid_diff = K.sigmoid(- ys_pred[:, 1:] + ys_pred[:, :1])
            # [batch, ..., n_samples]
            weighted_log_sigmoid_diff = probs_ys_pred[:, 1:] * K.log(sigmoid_diff)
            loss = - K.expand_dims(K.mean(K.sum(weighted_log_sigmoid_diff, axis=-1)), axis=0)
            return loss
        self.max_bpr_loss_layer = L.Lambda(max_bpr_loss)
        self.loss = self.max_bpr_loss_layer([self.sampled_targets, self.sampled_ratings])
        # self.loss = K.binary_crossentropy(self.sampled_targets, self.sampled_ratings)
        self.model = keras.Model([self.labels, self.sample_indices, self.output_ratings], self.loss)