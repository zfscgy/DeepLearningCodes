import tensorflow as tf
keras = tf.keras
L = keras.layers
A = keras.activations
K = keras.backend
from CoreKerasLayers.LossLayers import SampleLossLayer as _sampler



class MaxBPRLoss:
    """
    THis is the best loss function according to the paper
    'Recurrent Neural Networks for Top-k Recommendation'
    Note this class uses tf.gather_nd
    """
    def __init__(self, n_negative_samples, n_item_size):
        self.labels = L.Input([1], dtype='int32')
        self.sample_indices = L.Input([n_negative_samples], dtype='int32')
        self.output_ratings = L.Input([n_item_size])
        self.sampler = _sampler(n_negative_samples)
        self.sampled_targets, self.sampled_ratings = \
           self.sampler([self.labels, self.sample_indices, self.output_ratings])
        def max_bpr_loss(ys):
            ys_true, ys_pred = ys   # [batch, 1 + n_samples]
            probs_ys_pred = K.softmax(ys_pred)
            #                      [batch, n_samples]          [batch, 1]
            sigmoid_diff = K.sigmoid(K.transpose(K.transpose(- ys_pred[:, 1:]) + ys_pred[:, 0]))
            weighted_log_sigmoid_diff = probs_ys_pred[:, 1:] * K.log(sigmoid_diff)
            loss = - K.mean(K.sum(weighted_log_sigmoid_diff, -1), keepdims=True)
            return loss
        self.max_bpr_loss_layer = L.Lambda(max_bpr_loss)
        self.loss = self.max_bpr_loss_layer([self.sampled_targets, self.sampled_ratings])
        # self.loss = K.binary_crossentropy(self.sampled_targets, self.sampled_ratings)
        self.model = keras.Model([self.labels, self.sample_indices, self.output_ratings], self.loss)