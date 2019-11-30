import tensorflow as tf
k = tf.keras
L = k.layers
A = k.activations
K = k.backend
I = k.initializers
import numpy as np


class SampleOutputLayer(L.Layer):
    """
    The inputs of this layer is
    1. label [batch, .., 1]
    2. sample_indices [batch, .., n_samples]
    3. output_probalities [batch, .., item_size]
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
        y_preds, samples, labels = inputs
        # labels: [batch, ..]
        # samples: [batch, .., n_samples]
        # probs: [batch, .., item_size]
        labels = K.expand_dims(labels, -1)
        indices = K.concatenate([labels, samples], axis=-1)  # [batch, .., 1 + n_samples]
        indices = K.expand_dims(indices, -1)  # [batch, .., 1 + n_samples, 1]
        # [batch.., 1 + n_samples, 1], we have to expand because the last dim specifies index dim in tf.gather_nd
        gathered_probs = tf.gather_nd(y_preds, indices, batch_dims=1 + self.leading_dims)
        # gathered_probs: [batch, .., 1 + n_samples]
        targets = self.targets + K.zeros_like(gathered_probs)
        # [batch, .., 1 + n_samples]
        return targets, gathered_probs


class SampledCrossEntropy(L.Layer):
    def __init__(self, n_negative_samples, batch_size):
        super(SampledCrossEntropy, self).__init__()
        self.n_negative_samples = n_negative_samples
        self.batch_size = batch_size

    def build(self, input_shape):
        probs_shape, _ = input_shape
        self.n_items = probs_shape[-1]
        self.leading_shapes = probs_shape[1:-1]
        self.leading_dims = len(probs_shape) - 2
        self.sampler = SampleOutputLayer(self.n_negative_samples, self.leading_dims)
        super(SampledCrossEntropy, self).build(input_shape)

    def call(self, inputs, **kwargs):
        logits, labels = inputs
        sample_indices = K.random_uniform([self.batch_size] + self.leading_shapes + [self.n_negative_samples],
                                          0, self.n_items, dtype='int32')
        sampled_targets, sampled_logits = self.sampler([logits, sample_indices, labels])
        sampled_probs = K.softmax(sampled_logits)
        cross_entropy = K.categorical_crossentropy(sampled_targets, sampled_probs)
        return cross_entropy


class MaxBPRLoss(L.Layer):
    def __init__(self, n_negative_samples, batch_size):
        self.n_negative_samples = n_negative_samples
        self.batch_size = batch_size
        super(MaxBPRLoss, self).__init__()

    def build(self, input_shape):
        probs_shape, _ = input_shape
        self.n_items = probs_shape[-1]
        self.leading_shapes = probs_shape[1:-1]
        self.leading_dims = len(probs_shape) - 2
        self.sampler = SampleOutputLayer(self.n_negative_samples, self.leading_dims)
        super(MaxBPRLoss, self).build(input_shape)

    def call(self, inputs, **kwargs):
        logits, labels = inputs
        labels = labels[:, 0]
        sample_indices = K.random_uniform([self.batch_size] + self.leading_shapes + [self.n_negative_samples],
                                          0, self.n_items, dtype='int32')
        sampled_targets, sampled_logits = self.sampler([logits, sample_indices, labels])
        sampled_probs = K.softmax(sampled_logits)  # [batch, ..., n_samples + 1]
        sigmoid_diff = K.sigmoid(- sampled_probs[:, 1:] + sampled_probs[:, :1])
        # [batch, ..., n_samples]  This is the sigmoid difference of target probability and negative samples' probs
        weighted_log_sigmoid_diff = sampled_probs[:, 1:] * K.log(sigmoid_diff)
        loss = - K.sum(weighted_log_sigmoid_diff, axis=-1)
        return loss
