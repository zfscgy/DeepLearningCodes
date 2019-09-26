import tensorflow as tf
keras = tf.keras
L = keras.layers
Opts = keras.optimizers
Loss = keras.losses
M = keras.metrics
import numpy as np
from CoreKerasModels.CNNs import NextItemCNN
from Data.MovieLens.MovielensLoader import DataLoader
from Eval.Metrics import hit_ratio

movielens = DataLoader("1m")
movielens.generate_rating_history_seqs()
seq_len = 20
nextit_cnn = NextItemCNN(64, movielens.n_items, seq_len, [1, 2])
nextit_model = nextit_cnn.model
nextit_model.summary()
nextit_model.compile(Opts.Adam(0.0001), Loss.sparse_categorical_crossentropy)
n_rounds = 10000
batch_size = 32
for i in range(n_rounds):
    seqs = movielens.get_rating_history_test_batch(seq_len, batch_size)[0]
    nextit_model.train_on_batch(seqs[:, :-1], seqs[:, 1:])
    if i % 100 == 0:
        test_seqs = movielens.get_rating_history_test_batch(seq_len, 300)[0]
        pred_probs = nextit_model.predict(test_seqs[:, :-1])[:, -1, :]
        pred_max_20_vals = np.argpartition(-pred_probs, 20)[:, :20]  # [batch, 20]
        hr_20 = hit_ratio(pred_max_20_vals, test_seqs[:, -1])
        print("Hit ratio at 20:", hr_20)