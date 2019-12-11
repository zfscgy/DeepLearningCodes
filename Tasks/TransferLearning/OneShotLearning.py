import tensorflow as tf
from Keras.Models.Omniglot import OmniglotSiameseNetwork
from Data.OmniglotLoader import OmniglotLoader
keras = tf.keras
L = keras.layers
Opts = keras.optimizers
Loss = keras.losses
M = keras.metrics


def omniglot_experiment(hparams: dict=None, settings: dict=None, verbose=1):
    if hparams is None:
        hparams = dict()

    batch_size = hparams.get("batch_size", 64)
    learning_rate = hparams.get("learning_rate", 1e-3)

    if settings is None:
        settings = dict()

    n_rounds = settings.get("n_rounds", 5000)
    test_batch_size = settings.get("test_batch_size", 100),
    omniglot_loader = OmniglotLoader()
    model = OmniglotSiameseNetwork()
    model.compile(Opts.Adam(learning_rate), loss=Loss.binary_crossentropy, metrics=[M.binary_accuracy])
    model.summary()

    train_records = []
    test_records = []
    for i in range(n_rounds):
        images, labels = omniglot_loader.get_train_batch(batch_size)
        loss = model.train_on_batch(images, labels)
        train_records.append(loss)
        if verbose == 1:
            print("Iteration:", i, " Training loss & metric:", loss)
        if i % 100 == 0:
            eval_imgs, eval_labels = omniglot_loader.get_test_batch(test_batch_size)
            pred_labels = model.predict(eval_imgs)
            # print(pred_labels)
            acc = model.evaluate(eval_imgs, eval_labels)
            test_records.append((i, acc))
            if verbose == 1:
                print("Test loss & metric:", acc)
    return train_records, test_records


if __name__ == "__main__":
    omniglot_experiment()
