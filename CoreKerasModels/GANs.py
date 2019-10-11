import tensorflow as tf
L = tf.keras.layers
M = tf.keras.models
A = tf.keras.activations
Lo = tf.keras.losses
K = tf.keras.backend
import numpy as np


class DCGAN_128:
    def __init__(self, hidden_dim):
        generator = M.Sequential()
        generator.add(L.Dense(128 * 8 * 8, input_shape=[hidden_dim]))
        generator.add(L.Reshape([8, 8, 128]))
        generator.add(L.UpSampling2D())  # [8, 8, 128]
        generator.add(L.Conv2D(128, kernel_size=3, padding="same"))  # [16, 16, 128]
        generator.add(L.LayerNormalization())  # 4
        generator.add(L.ReLU())
        generator.add(L.UpSampling2D())  # [32, 32, 128]
        generator.add(L.Conv2D(64, kernel_size=5, padding="same"))   # [32, 32, 64]
        generator.add(L.LayerNormalization())  # 8
        generator.add(L.ReLU())
        generator.add(L.UpSampling2D())  # [64, 64, 128]
        generator.add(L.Conv2D(32, kernel_size=7, padding="same"))   # [64, 64, 32]
        generator.add(L.LayerNormalization())  # 12
        generator.add(L.ReLU())
        generator.add(L.UpSampling2D())  # [128, 128, 32]
        generator.add(L.Conv2D(3, kernel_size=3, padding="same", activation=A.sigmoid))   # [128, 128, 3]

        discriminator = M.Sequential()
        discriminator.add(L.Conv2D(32, kernel_size=5, strides=2, padding="same", input_shape=[128, 128, 3]))
        discriminator.add(L.LeakyReLU())
        # discriminator.add(L.Dropout(0.25))  # [64, 64, 32]
        discriminator.add(L.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # discriminator.add(L.BatchNormalization(epsilon=1e-5))  # 4
        discriminator.add(L.LeakyReLU())
        # discriminator.add(L.Dropout(0.25))  # [32, 32, 64]
        discriminator.add(L.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        discriminator.add(L.LayerNormalization())   # 8
        discriminator.add(L.LeakyReLU())    # [16, 16, 128]
        discriminator.add(L.Dropout(0.25))
        discriminator.add(L.Conv2D(256, kernel_size=3, strides=2, padding="same"))
        discriminator.add(L.LayerNormalization())   # 12
        discriminator.add(L.LeakyReLU())    # [8, 8, 256]
        discriminator.add(L.Dropout(0.25))
        discriminator.add(L.Conv2D(512, kernel_size=3, strides=2, padding="same"))
        discriminator.add(L.LeakyReLU())    # [4, 4, 512]
        discriminator.add(L.Flatten())
        discriminator.add(L.Dense(1, activation=A.sigmoid))
        self.model_gen = generator
        self.model_dis = discriminator

        self.adv_input = L.Input([hidden_dim])
        self.adv_output = discriminator(generator(self.adv_input))
        self.model_adversarial = M.Model(self.adv_input, self.adv_output)
        # self.model_adversarial = M.Sequential([generator, discriminator])
