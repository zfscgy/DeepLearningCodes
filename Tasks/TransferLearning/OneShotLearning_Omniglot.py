import tensorflow as tf
from CoreKerasModels.CNNs import OmniglotCNN
from CoreKerasModels.Distances import WeightedL1Distance
from Data.Omniglot.OmniglotLoader import OmniglotLoader
keras = tf.keras
L = keras.layers
Opts = keras.optimizers
Loss = keras.losses
M = keras.metrics

omniglot_cnn = OmniglotCNN()
weighted_l1_distance = WeightedL1Distance([4096])
cnn_model = omniglot_cnn.model
distance_model = weighted_l1_distance.model


input_1 = L.Input([105, 105, 1])
input_2 = L.Input([105, 105, 1])
features_1 = cnn_model(input_1)
features_2 = cnn_model(input_2)
distance = distance_model([features_1, features_2])
model = keras.Model(inputs=[input_1, input_2], outputs=distance)
model.summary()
model.compile(Opts.Adam(0.00006), loss=Loss.binary_crossentropy, metrics=[M.binary_accuracy])

omniglot_loader = OmniglotLoader("./Data/Omniglot/Omniglot Dataset", False, 32)
omniglot_loader.split_train_datasets()


n_rounds = 1000000
for i in range(n_rounds):
    images, labels = omniglot_loader.get_train_batch()

    loss = model.train_on_batch(images, labels)
    print("Iteration:", i, " Training loss & metric:", loss)
    if i % 1000 == 0:
        eval_imgs, eval_labels = omniglot_loader.get_test_batch(3, False, 5)
        pred_labels = model.predict(eval_imgs)
        for i in range(len(eval_labels)):
            print(eval_labels[i], pred_labels[i])
        acc = model.evaluate(eval_imgs, eval_labels)
        print("Test loss & metric:", acc)
