import tensorflow as tf
keras = tf.keras
L = keras.layers
Opts = keras.optimizers
Loss = keras.losses
M = keras.metrics

from CoreKerasModels.RNNs import GRU4Rec
from CoreKerasModels.CustomLosses import MaxBPRLoss
from Data.MovieLens.MovielensLoader import DataLoader
from Eval.Metrics import hit_ratio

movielens = DataLoader("1m")
movielens.generate_rating_history_seqs()
seq_len = 20
input_seq = L.Input([seq_len])
gru4rec = GRU4Rec(movielens.n_items, 64, seq_len)
gru4rec_model = gru4rec.model
out_softmax_layer = L.Softmax()
out_softmax = out_softmax_layer(gru4rec_model(input_seq))
max_bpr_loss = MaxBPRLoss(127, movielens.n_items)
max_bpr_loss_model = max_bpr_loss.model
