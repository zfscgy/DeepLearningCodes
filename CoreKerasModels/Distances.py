import tensorflow as tf
L = tf.keras.layers
M = tf.keras.models
A = tf.keras.activations
K = tf.keras.backend


class WeightedL1Distance:
    def __init__(self, input_shape):
        """

        :param input_shape:  [(batch), input_dim]
        """
        self.input_1 = L.Input(input_shape)
        self.input_2 = L.Input(input_shape)
        self.vec_1 = L.Flatten()(self.input_1)
        self.vec_2 = L.Flatten()(self.input_2)
        self.L1Distance_layer = L.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        self.l1_distance = self.L1Distance_layer([self.vec_1, self.vec_2])  # [input_dim]

        self.output_layer = L.Dense(1, activation=A.sigmoid)
        self.distance = self.output_layer(self.l1_distance)  # [1]
        self.model = M.Model(inputs=[self.input_1, self.input_2], outputs=self.distance)

    def get_models(self):
        return [self.model]
