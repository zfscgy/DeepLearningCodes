import tensorflow as tf
import numpy as np
from Keras.Components.LossLayers import MaxBPRLoss
k = tf.keras
L = tf.keras.layers
M = tf.keras.models
A = tf.keras.activations
Lo = tf.keras.losses
K = tf.keras.backend


class GRU4Rec(k.Model):
    def __init__(self, item_size, embedding_dim, sequence_length, gru_units):
        super(GRU4Rec, self).__init__()
        self.embedding_layer = L.Embedding(item_size, embedding_dim, input_length=sequence_length)
        self.gru_cell = L.GRUCell(gru_units)
        self.rnn = L.RNN(self.gru_cell, return_sequences=False)
        self.out_dense = L.Dense(item_size)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding_layer(inputs)  # [batch, seq_len, embedding_dim]
        x = self.rnn(x)  # [batch_ize, gru_units]
        x = self.out_dense(x)  # [batch_size, item_size]
        return x


class GRU4Rec_BPR(k.Model):
    def __init__(self, item_size, embedding_dim, sequence_length, gru_units,
                 n_negative_samples, batch_size):
        super(GRU4Rec_BPR, self).__init__()
        self.pred_model = GRU4Rec(item_size, embedding_dim, sequence_length, gru_units)
        self.bpr_loss = MaxBPRLoss(n_negative_samples, batch_size)
        self.predict([np.zeros([batch_size, sequence_length], dtype=np.int), np.zeros([batch_size], dtype=np.int)])

    def call(self, inputs, training=None, mask=None):
        xs, ys = inputs[0], inputs[1]
        pred_logits = self.pred_model(xs)
        loss = self.bpr_loss([pred_logits, ys])
        return loss
