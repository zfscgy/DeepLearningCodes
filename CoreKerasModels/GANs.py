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
        generator.add(L.Dense(512 * 8 * 8, input_shape=[hidden_dim]))
        generator.add(L.Reshape([8, 8, 512]))

        generator.add(L.Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
        # generator.add(L.BatchNormalization(momentum=0.5))
        generator.add(L.LeakyReLU())   # [16, 16, 256]

        generator.add(L.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
        # generator.add(L.BatchNormalization(momentum=0.5))
        generator.add(L.LayerNormalization())
        generator.add(L.LeakyReLU())   # [32, 32, 128]

        generator.add(L.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"))
        # generator.add(L.BatchNormalization(momentum=0.5))
        generator.add(L.LeakyReLU())  # [64, 64, 64]

        generator.add(L.Conv2DTranspose(32, kernel_size=5, strides=2, padding="same"))
        # generator.add(L.BatchNormalization(momentum=0.5))
        generator.add(L.LeakyReLU())  # [128, 128, 32]

        generator.add(L.Conv2D(3, kernel_size=3, padding="same", activation=A.tanh))   # [128, 128, 3]

        discriminator = M.Sequential()
        discriminator.add(L.Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=[128, 128, 3]))
        discriminator.add(L.LeakyReLU())  # [64, 64, 64]

        discriminator.add(L.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        # discriminator.add(L.BatchNormalization(momentum=0.5))
        discriminator.add(L.LeakyReLU())   # [32, 32, 128]

        discriminator.add(L.Conv2D(256, kernel_size=3, strides=2, padding="same"))
        # discriminator.add(L.BatchNormalization(momentum=0.5))
        discriminator.add(L.LayerNormalization())
        discriminator.add(L.LeakyReLU())    # [16, 16, 128]

        discriminator.add(L.Conv2D(512, kernel_size=3, strides=2, padding="same"))
        # discriminator.add(L.BatchNormalization(momentum=0.5))   # 12
        discriminator.add(L.LeakyReLU())    # [8, 8, 512]

        discriminator.add(L.Conv2D(1024, kernel_size=3, strides=2, padding="same"))
        # discriminator.add(L.BatchNormalization(momentum=0.5))
        discriminator.add(L.LeakyReLU())    # [4, 4, 1024]

        discriminator.add(L.Flatten())
        discriminator.add(L.Dropout(0.2))
        discriminator.add(L.Dense(1, activation=A.sigmoid))

        self.model_gen = generator
        self.model_dis = discriminator

        self.adv_input = L.Input([hidden_dim])
        self.adv_output = discriminator(generator(self.adv_input))
        self.model_adversarial = M.Model(self.adv_input, self.adv_output)
        # self.model_adversarial = M.Sequential([generator, discriminator])
