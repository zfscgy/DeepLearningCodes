import tensorflow as tf
L = tf.keras.layers
A = tf.keras.activations
K = tf.keras.backend
M = tf.keras.models
Ops = tf.keras.optimizers
Losses = tf.keras.losses
import numpy as np


def test_nested_layers():
    class MyDenseLayer(L.Layer):
        def __init__(self, units):
            self.units = units
            self.dense_layer = L.Dense(self.units)
            super(MyDenseLayer, self).__init__()

        def build(self, input_shape):
            self.dense_layer.build(input_shape)
            # Must add following two lines, otherwise the nested dense layer won't be trained
            # self._trainable_weights += self.dense_layer.trainable_weights
            # self._non_trainable_weights += self.dense_layer.non_trainable_weights
            super(MyDenseLayer, self).build(input_shape)
            print(self._trainable_weights)

        def call(self, inputs, **kwargs):
            print("Layer called")
            return self.dense_layer.call(inputs)

        def compute_output_shape(self, input_shape):
            return list(input_shape[:-1]).append(self.units)

    class RandomLayer(L.Layer):
        def call(self, inputs):
            return K.cast(K.random_uniform([1], 0, 10), 'int32')

    x1s = np.random.uniform(-1, 1, [1000, 1])
    x2s = np.random.uniform(0, 1, [1000, 1])
    zs = x1s + x2s
    xs = np.hstack([x1s, x2s])


    # Test nested layers
    model = tf.keras.Sequential()
    model.add(MyDenseLayer(1))
    model.add(RandomLayer())
    model.compile(Ops.SGD(0.1), Losses.mean_squared_error)
    model.fit(xs, zs)

def test_look_up_model():
    input_1 = L.Input([10])
    input_2 = L.Input([2, 1], dtype='int32')
    out = tf.gather_nd(input_1, input_2, batch_dims=1)
    model = M.Model([input_1, input_2], out)
    print(model.predict(([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], [[[0], [1]]])))

def test_2_inputs_layer():
    class DualInputLayer(L.Layer):
        def __init__(self):
            super(DualInputLayer, self).__init__()
        def build(self, input_shape):
            shape1, shape2 = input_shape
            print(shape1, shape2)
        def call(self, inputs):
            return L.concatenate(inputs)
    input_1 = L.Input([1])
    input_2 = L.Input([1])
    output_layer = DualInputLayer()
    output = output_layer([input_1, input_2])
    model = M.Model([input_1, input_2], output)
    print(model.predict([[1], [2]]))

def test_trainable():
    model_1 = M.Sequential()
    model_1.add(L.Dense(1, input_shape=[10]))
    model_3 = M.Sequential()
    model_3.add(model_1)
    model_4 = M.Sequential()
    model_4.add(model_1)

    model_3.compile(Ops.SGD(0.1), Losses.mean_squared_error)
    model_4.compile(Ops.SGD(0.1), Losses.mean_squared_error)
    xs = np.random.uniform(0, 1, [100, 10])
    ys = np.sum(xs, axis=1, keepdims=True)

    model_3.predict(xs)
    print(model_3.get_weights())
    # model_1.trainable = False
    model_3. train_on_batch(xs, ys)
    print(model_3.get_weights())
    model_3.trainable = False
    model_4.train_on_batch(xs, ys)
    print(model_4.get_weights())
    model_1.trainable = True
    model_4.train_on_batch(xs, ys)
    print(model_4.get_weights())
    '''
    inputs = L.Input(10)
    output = model_1(inputs, training=False)
    model_5 = M.Model(inputs, output)
    model_5.compile(Ops.SGD(0.1), Losses.mean_squared_error)
    print(model_5.get_weights())
    model_5.fit(xs, ys)
    print(model_5.get_weights())
    '''

def test_batch_normalization():
    def loss_y(y_true, y_pred):
        return K.var(y_pred)
    x1 = np.random.normal(0, 1, [10, 1])
    x2 = np.random.normal(0, 2, [10, 1])
    xs = np.hstack([x1, x2])
    model = M.Sequential([L.BatchNormalization(input_shape=[2])])
    # model.trainable = False
    model.compile(Ops.SGD(0), loss_y)
    ys = model.train_on_batch(xs, xs)
    print(ys)
    x3 = np.random.normal(0, 100, [10, 1])
    x4 = np.random.normal(0, 10, [10, 1])
    test_xs = np.hstack([x3, x4])
    for i in range(100):
        ys0 = model.train_on_batch(test_xs, test_xs)
    ys1 = model.test_on_batch(test_xs, test_xs)
    ys2 = model.train_on_batch(test_xs, test_xs)
    ys3 = model.test_on_batch(test_xs, test_xs)
    for i in range(100):
        ys4 = model.test_on_batch(test_xs, test_xs)
    print(ys0, ys1, ys2, ys3, ys4)

test_batch_normalization()