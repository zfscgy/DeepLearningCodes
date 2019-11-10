import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Eval.Metrics import binary_cross_entropy
k = tf.keras
A = k.activations
L = k.layers
Op = k.optimizers
Lo = k.losses
from Data.Datasets.SimulatedData.TwoDSimData import AxeDataGenerator
data_generator = AxeDataGenerator(train_size=200)
train_data = data_generator.data[:data_generator.train_index]
class_0_data = np.array([data[:2] for data in train_data if data[2] == 0])
class_1_data = np.array([data[:2] for data in train_data if data[2] == 1])
plt.plot(class_0_data[:, 0], class_0_data[:, 1], 'o')
plt.plot(class_1_data[:, 0], class_1_data[:, 1], 'x')
plt.show()

def vanilla_classify():

    '''
    data_grid = np.array([[[0.0, 0.0]] * 100] * 100)
    for i in range(100):
        for j in range(100):
            data_grid[i, j] = [i * 0.02 - 1, j * 0.02 - 1]
    data_grid = data_grid.reshape([100 * 100, 2])
    '''
    classifier = k.Sequential()
    classifier.add(L.Dense(10, input_shape=[2]))
    classifier.add(L.LeakyReLU())
    classifier.add(L.Dense(1, activation=A.sigmoid))
    classifier.summary()
    classifier.compile(Op.Adam(), Lo.binary_crossentropy)
    batch_size = 32
    losses = []
    for i in range(15000):
        xs, ys = data_generator.get_train_batch(batch_size)
        loss = classifier.train_on_batch(xs, ys)
        if i % 100 == 0:
            print("Round {:4d} loss {:.5f}".format(i, loss))
            test_xs, test_ys = data_generator.get_test_data()
            preds = classifier.predict(test_xs)[:, 0]
            test_loss = classifier.evaluate(test_xs, test_ys)
            losses.append(test_loss)
            print("Test loss {:.5f}".format(test_loss))


    preds = np.round(preds)
    c0grid = test_xs[preds == 0]
    c1grid = test_xs[preds == 1]
    plt.plot(c0grid[:, 0], c0grid[:, 1], 'o')
    plt.plot(c1grid[:, 0], c1grid[:, 1], 'x')
    plt.show()
    c0grid = test_xs[test_ys[:, 0] == 0]
    c1grid = test_xs[test_ys[:, 0] == 1]
    plt.plot(c0grid[:, 0], c0grid[:, 1], 'o')
    plt.plot(c1grid[:, 0], c1grid[:, 1], 'x')
    plt.show()
    return losses

def acgan_classify():
    generator = k.Sequential()
    generator.add(L.Dense(20, input_shape=[3]))
    generator.add(L.LeakyReLU())
    generator.add(L.Dense(10, A.tanh))
    generator.add(L.Dense(2))

    def dis_loss(y_true, y_pred):
        return Lo.binary_crossentropy(y_true[:, :1], y_pred[:, :1]) + \
            Lo.binary_crossentropy(y_true[:, 1:], y_pred[:, 1:])

    def gen_loss(y_true, y_pred):
        return - Lo.binary_crossentropy(y_true[:, :1], y_pred[:, :1]) + \
            Lo.binary_crossentropy(y_true[:, 1:], y_pred[:, 1:])



    discriminator = k.Sequential()
    discriminator.add(L.Dense(10, input_shape=[2], activation=A.tanh))
    discriminator.add(L.Dense(2, activation=A.sigmoid))
    discriminator.compile(Op.Adam(), dis_loss)

    discriminator.trainable = False
    gan = k.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(Op.Adam(), gen_loss)
    losses = []
    batch_size = 32
    for i in range(15000):
        gen_labels = np.random.randint(0, 2, [batch_size, 1])
        gen_noise = np.random.normal(0, 1, [batch_size, 2])
        generated_xs = generator.predict(np.hstack([gen_labels, gen_noise]))
        generated_ys = np.hstack([np.zeros([batch_size, 1]), gen_labels])
        real_xs, real_ys = data_generator.get_train_batch(batch_size)
        real_ys = np.hstack([np.ones([batch_size, 1]), real_ys])

        loss_fake = discriminator.train_on_batch(generated_xs, generated_ys)
        loss_real = discriminator.train_on_batch(real_xs, real_ys)

        gen_labels = np.random.randint(0, 2, [batch_size, 1])
        gen_noise = np.random.normal(0, 1, [batch_size, 2])
        generated_ys = np.hstack([np.zeros([batch_size, 1]), gen_labels])
        loss_gan = gan.train_on_batch(np.hstack([gen_labels, gen_noise]), generated_ys)
        if i % 100 == 0:
            print("Round {:4d} Loss real {:.5f}, Loss fake {:.5f}, Loss gan {:.5f}".
                  format(i, loss_fake, loss_real, loss_gan))
            test_xs, test_ys = data_generator.get_test_data()
            preds = discriminator.predict(test_xs)[:, 1:]
            test_loss = binary_cross_entropy(test_ys, preds)
            losses.append(test_loss)
            print("Test loss {:.5f}".format(test_loss))

    preds = np.round(preds[:, 0])
    c0grid = test_xs[preds == 0]
    c1grid = test_xs[preds == 1]
    plt.plot(c0grid[:, 0], c0grid[:, 1], 'o')
    plt.plot(c1grid[:, 0], c1grid[:, 1], 'x')
    plt.show()
    c0grid = test_xs[test_ys[:, 0] == 0]
    c1grid = test_xs[test_ys[:, 0] == 1]
    plt.plot(c0grid[:, 0], c0grid[:, 1], 'o')
    plt.plot(c1grid[:, 0], c1grid[:, 1], 'x')
    plt.show()
    return losses

loss1 = vanilla_classify()
loss2 = acgan_classify()
plt.plot(loss1, 'x')
plt.plot(loss2, 'o')
plt.show()