import numpy as np

def binary_cross_entropy(targets, predictions, epsilon=1e-9):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    ce = - np.mean(np.log(predictions) * targets + np.log(1 - predictions) * (1 - targets))
    return ce


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """

    :param y_pred: shape [batch, pred_len]
    :param y_true: shape [batch]
    :return:
    """
    batch_size = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    y_pred_class = np.argmax(y_pred, axis=1)
    hit_array = np.equal(y_pred_class, y_true).astype(np.float)
    return np.sum(hit_array) / batch_size

def cross_entropy(predictions, targets, epsilon=1e-9):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    ce = - np.mean(np.log(predictions) * targets)
    return ce


def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray):
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


def discounted_cumulative_gain(y_true: np.ndarray, y_pred: np.ndarray):
    batch_size = y_pred.shape[0]
    pred_len = y_pred.shape[1]
    y_true = np.broadcast_to(y_true, [pred_len, batch_size]).transpose()
    hit_array = np.equal(y_pred, y_true).astype(np.float)  # [batch, pred_len]
    scores = np.hstack((hit_array[:, :1], hit_array[:, 1:] / np.log2(np.arange(2, pred_len + 1))))
    return np.mean(np.sum(scores, axis=1))


def oneshot_accuracy(y_pred:np.ndarray, support_size:int=2):
    """
    :param y_pred: shape [batch_size * support_size, 1]
    Assume in each batch, the 1 label is in the first place
    i.e. y_true must be [1 0 0 ...(0 for support_size - 1) 1 0 0 ... 1 0 0 ... ...]
    :return:
    """
    len = y_pred.shape[0]
    ys = y_pred.reshape([int(len/support_size), support_size])
    max_idx = np.argmax(ys, axis=1)
    return np.mean(max_idx == 0)