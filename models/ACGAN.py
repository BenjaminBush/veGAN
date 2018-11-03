from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.layers.noise import GaussianNoise
from utils.Minibatch import MinibatchDiscrimination


class ACGAN(object):
    def __init__(self, latent_size=110, class_num=10):
        self.latent_size = latent_size
        self.class_num = class_num
        self.generator = self.build_generator(self.latent_size)
        self.discriminator = self.build_discriminator(self.class_num) 

    def build_generator(self, latent_size):
        # we will map a pair of (z, L), where z is a latent vector and L is a
        # label drawn from P_c, to image space (..., 3, 32, 32)
        cnn = Sequential()
        cnn.add(Dense(384 * 4 * 4, input_dim=latent_size, activation='relu',
                      kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(Reshape((384, 4, 4)))

        cnn.add(Conv2DTranspose(192, kernel_size=5, strides=2, padding='same', activation='relu',
                                kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(BatchNormalization())

        cnn.add(Conv2DTranspose(96, kernel_size=5, strides=2, padding='same', activation='relu',
                                kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(BatchNormalization())

        cnn.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh',
                                kernel_initializer='glorot_normal', bias_initializer='Zeros'))

        # this is the z space commonly refered to in GAN papers
        latent = Input(shape=(latent_size,))

        # this will be our label
        image_class = Input(shape=(1,), dtype='int32')

        # 10 classes in CIFAR-10
        cls = Flatten()(Embedding(10, latent_size,
                                  embeddings_initializer='glorot_normal')(image_class))

        # hadamard product between z-space and a class conditional embedding
        h = layers.multiply([latent, cls])

        fake_image = cnn(h)

        return Model([latent, image_class], fake_image)


    def build_discriminator(self, class_num):
        # build a relatively standard conv net, with LeakyReLUs as suggested in
        # the reference paper
        cnn = Sequential()

        cnn.add(GaussianNoise(0.05, input_shape=(3, 32, 32)))  # Add this layer to prevent D from overfitting!

        cnn.add(Conv2D(16, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.5))

        cnn.add(Conv2D(32, kernel_size=3, strides=1, padding='same',
                       kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(BatchNormalization())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.5))

        cnn.add(Conv2D(64, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(BatchNormalization())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.5))

        cnn.add(Conv2D(128, kernel_size=3, strides=1, padding='same',
                       kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(BatchNormalization())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.5))

        cnn.add(Conv2D(256, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(BatchNormalization())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.5))

        cnn.add(Conv2D(512, kernel_size=3, strides=1, padding='same',
                       kernel_initializer='glorot_normal', bias_initializer='Zeros'))
        cnn.add(BatchNormalization())
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Dropout(0.5))

        cnn.add(Flatten())

        cnn.add(MinibatchDiscrimination(50, 30))

        image = Input(shape=(3, 32, 32))

        features = cnn(image)

        # first output (name=generation) is whether or not the discriminator
        # thinks the image that is being shown is fake, and the second output
        # (name=auxiliary) is the class that the discriminator thinks the image
        # belongs to.
        fake = Dense(1, activation='sigmoid', name='generation',
                     kernel_initializer='glorot_normal', bias_initializer='Zeros')(features)
        aux = Dense(class_num, activation='softmax', name='auxiliary',
                    kernel_initializer='glorot_normal', bias_initializer='Zeros')(features)

        return Model(image, [fake, aux])