import tensorflow as tf
import numpy as np
from Keras.Models.Gru4Rec import GRU4Rec_BPR
from Eval.Metrics import hit_ratio, discounted_cumulative_gain as dcg
from Data import RatingSeqDataLoader
keras = tf.keras
L = keras.layers
Opts = keras.optimizers
Loss = keras.losses
M = keras.metrics


def gru4rec_experiment(hparams:dict = None, settings:dict = None, verbose=1):
    if hparams is None:
        hparams = dict()
    batch_size = hparams.get("batch_size", 32)
    learning_rate = hparams.get("learning_rate", 1e-2)
    seq_len = hparams.get("seq_len", 8)
    item_feature_dim = hparams.get("item_feature_dim", 16)
    n_negative_samples = hparams.get("n_negative_samples", 127)
    gru_units = hparams.get("gru_units", 64)
    if settings is None:
        settings = dict()
    n_rounds = hparams.get("n_rounds", 1000)
    dataset = settings.get("dataset", "m-1m")
    data_loader = RatingSeqDataLoader(dataset, seq_len=seq_len, label_mode="last")
    gru4rec = GRU4Rec_BPR(data_loader.n_items, item_feature_dim, seq_len, gru_units, n_negative_samples, batch_size)
    gru4rec.compile(Opts.Adam(learning_rate), lambda y_true, y_pred: y_pred)
    gru4rec.summary()
    for i in range(n_rounds):
        if i % 100 == 0:
            xs, ys = data_loader.get_test_batch()
            ys = ys[:, -1]
            # Using a array to store top50 candidates for each test sequence
            pred_max_vals = np.zeros([xs.shape[0], 50])
            # Using a for-loop to predict, prevent out-of-memory, predict every 1000 sequences
            for j in range(0, xs.shape[0], 1000):
                end = min(j + 1000, xs.shape[0])
                pred_probs = gru4rec.pred_model.predict(xs[j: end, :])
                pred_max_vals[j:end, :] = np.argpartition(-pred_probs, 10)[:, :50]
            hr_10 = hit_ratio(ys, pred_max_vals[:, :10])
            dcg_10 = dcg(ys, pred_max_vals[:, :10])
            hr_50 = hit_ratio(ys, pred_max_vals[:, :50])
            dcg_50 = dcg(ys, pred_max_vals[:, :50])
            if verbose == 1:
                print("round {}\n10: hr {:.4f} dcg {:.4f} \n50: hr {:.4f} dcg {:.4f}".
                      format(i, hr_10, dcg_10, hr_50, dcg_50))

        xs, ys = data_loader.get_train_batch(batch_size)
        loss = gru4rec.train_on_batch([xs, ys], np.zeros([batch_size]))


if __name__ == "__main__":
    gru4rec_experiment()
