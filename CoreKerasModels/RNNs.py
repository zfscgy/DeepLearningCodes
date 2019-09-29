import tensorflow as tf
L = tf.keras.layers
M = tf.keras.models
A = tf.keras.activations
Lo = tf.keras.losses
K = tf.keras.backend

class GRU4Rec:
    def __init__(self, item_size, embedding_dim, seq_len,
                 dropout_rate=0.0, gru_units=50):
        self.inputs = L.Input(shape=(seq_len,))    # [batch_size, seq_len]
        self.embedding_layer = L.Embedding(item_size, embedding_dim, input_length=seq_len)
        self.embedding_seq = self.embedding_layer(self.inputs)  # [batch_size, seq_len, feature_dim]
        self.gru_cell = L.GRUCell(gru_units)
        self.gru_rnn = L.RNN(self.gru_cell, return_sequences=True)
        self.gru_out = self.gru_rnn(self.embedding_seq)  # [batch_size, seq_len, gru_units]
        self.gru_dropout_layer = L.Dropout(dropout_rate)
        self.dropout_out = self.gru_dropout_layer(self.gru_out)
        self.out_ratings = L.Dense(item_size, activation=A.linear)
        self.model = M.Model(self.inputs, self.out_ratings)