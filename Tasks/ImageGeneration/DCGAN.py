from CoreKerasModels.GANs import DCGAN_128
from Data.AnimeFaces.AnimeFacesLoader import AnimeFacesLoader
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
Opt = tf.keras.optimizers
Lo = tf.keras.losses



hidden_dim = 100

dcgan = DCGAN_128(hidden_dim)
data_loader = AnimeFacesLoader([128, 128])

batch_size = 32
n_rounds = 40000

dis_model = dcgan.model_dis
gen_model = dcgan.model_gen
adv_model = dcgan.model_adversarial
gen_model.summary()
adv_model.summary()



dis_model.compile(Opt.Adam(0.0001), Lo.binary_crossentropy)
adv_model.compile(Opt.Adam(0.0001), Lo.binary_crossentropy)

layer_outputs = [layer.output for layer in dis_model.layers]
visual_model = tf.keras.Model(dis_model.input, layer_outputs)

for rounds in range(n_rounds):
    # Get output images
    if rounds % 100 == 0 and rounds > 0:
        noise = np.random.uniform(0, 1, [16, hidden_dim])
        tiled_images = np.zeros([4*128, 4*128, 3]).astype(np.uint8)
        generated_imgs = gen_model.predict(noise)
        generated_imgs *= 256
        generated_imgs = generated_imgs.astype(np.uint8)
        for i in range(16):
            tiled_images[int(i / 4)*128: int(i / 4)*128 + 128,
                         int(i % 4)*128: int(i % 4)*128 + 128, :] = generated_imgs[i, :, :, :]
        Image.fromarray(tiled_images).save("Output/DCGAN/" + "rounds_{0}.jpg".format(rounds))


    '''
        layer_visualization = visual_model.predict(generated_imgs[:1])
        for i in range(len(layer_visualization)):
            plt.imshow(layer_visualization[i][0, :, :, 0])
            plt.show()
    '''

    # train discriminator on real images
    real_imgs = data_loader.get_batch(batch_size)
    real_ys = np.ones([batch_size, 1])
    # train discriminator on fake images
    noise = np.random.uniform(-1, 1, [batch_size, hidden_dim])
    # print(noise)
    fake_ys = np.zeros([batch_size, 1])
    fake_imgs = gen_model.predict(noise)
    imgs = np.concatenate([real_imgs, fake_imgs], axis=0)
    ys = np.concatenate([real_ys, fake_ys], axis=0)
    loss_dis = dis_model.train_on_batch(imgs, ys)
    print("Round {}, Loss dis:{:.4f}".format(rounds, loss_dis))

    # train generator
    dis_model.trainable = False
    noise = np.random.uniform(-1, 1, [batch_size, hidden_dim])
    # print(noise)
    fake_ys = np.ones([batch_size, 1])
    # loss_gen_test = tf.keras.backend.eval(tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.keras.backend.variable(fake_ys), adv_model(noise))))

    loss_gen = adv_model.train_on_batch(noise, fake_ys)
    loss_gen_test = adv_model.test_on_batch(noise, fake_ys)
    print(loss_gen_test)
    print("Round {}, Loss gen:{:.4f}".format(rounds, loss_gen))
    # reset
    dis_model.trainable = True
