from Keras.CoreKerasModels.GANs import DCGAN_128
from Data.Datasets.AnimeFaces.AnimeFacesLoader import AnimeFacesLoader
import numpy as np
import tensorflow as tf
from PIL import Image

Opt = tf.keras.optimizers
Lo = tf.keras.losses

'''
    Some training tips:
    1. For deep networks, learning rate should be rather small
    2. In my experiments, batch normalization always causes unstable training, more severely, since batch normalization 
    behaves differently between training and testing, generator and discriminator's losses can be zero at same time. 
    Since we train generator using discriminator's test mode and train discriminator with generator's test mode, 
    generators can generate shit1(training mode) for discriminator when training and discriminator get shit2(testing
    mode), so generator can learn to distinguish shit2 but not shit1, generator learn to generate shit1 to fool test
    mode discriminator.
    Even if removing all batch normalization layers in generator, the generator can still learn some shit to fool 
    test-mode discriminator.
    Notice: in test-mode, the batch normalization layer uses moving average of mean and var, but in train-mode, it 
    compute mean and var within the batch.
    3. The dense layer in both generator and discriminator should link small-size filters (4x4), if the filters are too
    large, we should add a conv(deconv) layer to it to make it smaller.
    4. Discriminator always outperforms generator, so I train generator 3 times every step.
    5. Also used label smoothing for real images(not sure whether it's useful)
'''



hidden_dim = 100
dcgan = DCGAN_128(hidden_dim)
data_loader = AnimeFacesLoader([128, 128])

batch_size = 48
n_rounds = 40000

dis_model = dcgan.model_dis
gen_model = dcgan.model_gen
adv_model = dcgan.model_adversarial
gen_model.summary()
adv_model.summary()




dis_model.compile(Opt.Adam(0.00002), Lo.binary_crossentropy)
dis_model.trainable = False
adv_model.compile(Opt.Adam(0.00002), Lo.binary_crossentropy)

layer_outputs = [layer.output for layer in dis_model.layers]
visual_model = tf.keras.Model(dis_model.input, layer_outputs)



for rounds in range(n_rounds):
    # Get output images
    if rounds % 100 == 0 and rounds > 0:
        noise = np.random.normal(0, 1, [16, hidden_dim])
        tiled_images = np.zeros([4*128, 4*128, 3]).astype(np.uint8)
        generated_imgs = gen_model.predict(noise)
        generated_imgs *= 128
        generated_imgs += 128
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

    # train discriminator on real & fake images
    real_imgs = data_loader.get_batch(batch_size)
    real_ys = np.ones([batch_size, 1]) + np.random.uniform(-0.1, 0, [batch_size, 1])
    noise = np.random.normal(0, 1, [batch_size, hidden_dim])
    fake_ys = np.zeros([batch_size, 1])
    fake_imgs = gen_model.predict(noise)
    # imgs = np.concatenate([real_imgs, fake_imgs], axis=0)
    # ys = np.concatenate([real_ys, fake_ys], axis=0)


    loss_dis_real = dis_model.train_on_batch(real_imgs, real_ys)
    loss_dis_fake = dis_model.train_on_batch(fake_imgs, fake_ys)
    print("Round {}, Loss dis real:{:.4f} fake:{:.4f}".format(rounds, loss_dis_real, loss_dis_fake))
    loss_dis_test_real = dis_model.train_on_batch(real_imgs, real_ys)
    loss_dis_test_fake = dis_model.train_on_batch(fake_imgs, fake_ys)
    print(loss_dis_test_real, loss_dis_test_fake)
    for _ in range(3):
        noise = np.random.normal(0, 1, [batch_size, hidden_dim])
        fake_ys = np.ones([batch_size, 1])

        loss_gen = adv_model.train_on_batch(noise, fake_ys)
        print("Round {}, Loss gen:{:.4f}".format(rounds, loss_gen))
        loss_gen_test = adv_model.test_on_batch(noise, fake_ys)
        print(loss_gen_test)
