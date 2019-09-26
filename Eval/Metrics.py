import numpy as np


def hit_ratio(y_pred: np.ndarray, y_true: np.ndarray):
    """

    :param y_pred: shape [batch, pred_len]
    :param y_true: shape [batch]
    :return:
    """
    batch_size = y_pred.shape[0]
    pred_len = y_pred.shape[1]
    y_true = np.broadcast_to(y_true, [pred_len, batch_size]).transpose()
    hit_array = np.equal(y_pred, y_true).astype(np.float)
    return np.sum(hit_array) / batch_size
