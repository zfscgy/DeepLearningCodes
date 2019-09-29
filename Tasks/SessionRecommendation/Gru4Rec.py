import tensorflow as tf
keras = tf.keras
L = keras.layers
Opts = keras.optimizers
Loss = keras.losses
M = keras.metrics
import numpy as np
from CoreKerasModels.RNNs import GRU4Rec
from CoreKerasModels.CustomLosses import MaxBPRLoss
from Data.MovieLens.MovielensLoader import DataLoader
from Eval.Metrics import hit_ratio
from Utils.Sampler import Sampler


movielens = DataLoader("1m")
movielens.generate_rating_history_seqs()

seq_len = 20
n_negative_samples = 127

input_seq = L.Input([seq_len])

gru4rec = GRU4Rec(movielens.n_items, 64, seq_len)
gru4rec_model = gru4rec.model
gru_out = gru4rec_model(input_seq)  # [batch, n_items]

out_softmax_layer = L.Softmax()
out_softmax = out_softmax_layer(gru_out)  # [batch, n_items]

max_bpr_loss = MaxBPRLoss(n_negative_samples, movielens.n_items)
max_bpr_loss_model = max_bpr_loss.model

model_pred = keras.Model(input_seq, out_softmax)
# model_pred.compile(Opts.Adam(0.0001), Loss.mse)
input_labels = L.Input([1], dtype='int32')
negative_samples = L.Input([n_negative_samples], dtype='int32')
max_bpr_loss_output = max_bpr_loss_model([input_labels, negative_samples, gru_out])

model_train = keras.Model([input_seq, input_labels, negative_samples], max_bpr_loss_output)
model_train.compile(Opts.SGD(), lambda y_true, y_pred: y_pred)


n_rounds = 10000
batch_size = 32
sampler = Sampler(movielens.n_items, n_negative_samples)
for i in range(n_rounds):
    if i % 100 == 0:
        test_seqs = movielens.get_rating_history_test_batch(seq_len, 300)[0]
        pred_probs = model_pred.predict(test_seqs[:, :-1])
        pred_max_20_vals = np.argpartition(-pred_probs, 20)[:, :20]  # [batch, 20]
        hr_20 = hit_ratio(pred_max_20_vals, test_seqs[:, -1])
        print("Hit ratio at 20:", hr_20)
    seqs = movielens.get_rating_history_test_batch(seq_len, batch_size)[0]
    model_train.train_on_batch([seqs[:, :-1], seqs[:, -1], sampler.get_samples(batch_size)], np.zeros(batch_size))

