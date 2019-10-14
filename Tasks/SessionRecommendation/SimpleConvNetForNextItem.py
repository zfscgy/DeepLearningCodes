import tensorflow as tf
keras = tf.keras
L = keras.layers
Opts = keras.optimizers
Loss = keras.losses
M = keras.metrics
import numpy as np
from CoreKerasModels.CNNs import NextItemCNN
from CoreKerasModels.CustomLosses import SampledCrossEntropy, MaxBPRLoss
from Data.MovieLens.MovielensLoader import DataLoader
from Eval.Metrics import hit_ratio, discounted_cumulative_gain as dcg
from Utils.Sampler import Sampler

movielens = DataLoader("1m-u10i5")
seq_len = 8
n_negative_samples = 100
movielens.generate_rating_history_seqs(min_len=seq_len+2)

nextit_cnn = NextItemCNN(64, movielens.n_items, seq_len, [(1, 3), (2, 2)])
nextit_model_train = nextit_cnn.model_logits
nextit_model_test = nextit_cnn.model
input_seq = L.Input([seq_len])
cnn_out = nextit_model_train(input_seq)
#  [batch, seq_len, item_size]
input_labels = L.Input([seq_len], dtype='int32')  # [batch, seq_len]
sampler = Sampler(movielens.n_items, n_negative_samples)
negative_samples = L.Input([seq_len, n_negative_samples], dtype='int32')
sampled_loss_class = SampledCrossEntropy(n_negative_samples, movielens.n_items, leading_shape=[seq_len])
sampled_loss_model = sampled_loss_class.model
sampled_loss = sampled_loss_model([input_labels, negative_samples, cnn_out])
# maxbpr_loss_class = MaxBPRLoss(100, movielens.n_items, [seq_len])
# maxbpr_loss_model = maxbpr_loss_class.model
# sampled_loss = maxbpr_loss_model([input_labels, negative_samples, cnn_out])
model_train = keras.Model([input_seq, input_labels, negative_samples], sampled_loss)
model_train.summary()

nextit_model_train.summary()
model_train.compile(Opts.Adam(), lambda y_true, y_pred: y_pred)
# nextit_model_test.compile(Opts.Adam(0.0006), Loss.sparse_categorical_crossentropy)
n_rounds = 10000
batch_size = 32
for i in range(n_rounds):
    if i % 100 == 0:
        test_seqs = movielens.get_test_batch_from_all_user(seq_len)[0]
        pred_probs = nextit_model_test.predict(test_seqs[:, :-1])[:, -1, :]
        pred_max_vals = np.argpartition(-pred_probs, 50)[:, :50]  # [batch, 20]
        hr_10 = hit_ratio(test_seqs[:, -1], pred_max_vals[:, :10])
        dcg_10 = dcg(test_seqs[:, -1], pred_max_vals[:, :10])
        hr_50 = hit_ratio(test_seqs[:, -1], pred_max_vals[:, :50])
        dcg_50 = dcg(test_seqs[:, -1], pred_max_vals[:, :50])
        print("round {}\n10: hr {:.4f} dcg {:.4f} \n50: hr {:.4f} dcg {:.4f}".format(i, hr_10, dcg_10, hr_50, dcg_50))
    seqs = movielens.get_train_batch_from_all_user(seq_len, batch_size)[0]
    loss = model_train.train_on_batch(
        [seqs[:, :-1], seqs[:, 1:],
         sampler.get_samples(seq_len * batch_size).reshape([batch_size, seq_len, n_negative_samples])],
        np.zeros([batch_size, 1]))
#    loss = nextit_model_test.train_on_batch(seqs[:, :-1], seqs[:, 1:])
#    print(loss)

