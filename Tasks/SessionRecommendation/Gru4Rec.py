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
from Eval.Metrics import hit_ratio, discounted_cumulative_gain as dcg
from Utils.Sampler import Sampler


movielens = DataLoader("1m-u10i5")


seq_len = 8
movielens.generate_rating_history_seqs(min_len=seq_len+2)
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
model_train.summary()
model_train.compile(Opts.Adam(), lambda y_true, y_pred: y_pred)


n_rounds = 10000
batch_size = 32
sampler = Sampler(movielens.n_items, n_negative_samples)
for i in range(n_rounds):
    if i % 100 == 0:
        test_seqs = movielens.get_test_batch_from_all_user(seq_len)[0]
        pred_probs = model_pred.predict(test_seqs[:, :-1])
        pred_max_10_vals = np.argpartition(-pred_probs, 10)[:, :10]  # [batch, 20]
        hr_10 = hit_ratio(test_seqs[:, -1], pred_max_10_vals)
        dcg_10 = dcg(test_seqs[:, -1], pred_max_10_vals)
        print("Round:{}\t\t\tHR_10:{:.4f} NDCG_10:{:.4f}".format(i, hr_10, dcg_10))
    seqs = movielens.get_train_batch_from_all_user(seq_len, batch_size)[0]
    model_train.train_on_batch([seqs[:, :-1], seqs[:, -1], sampler.get_samples(batch_size)], np.zeros(batch_size))
