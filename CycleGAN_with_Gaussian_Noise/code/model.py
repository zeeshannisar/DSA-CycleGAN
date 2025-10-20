import h5py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, MaxPooling2D, Cropping2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, LayerNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
import custum_layers
from scipy.stats import wasserstein_distance

from utils import data_utils

# Discriminator takes half of the loss (practical advice from the paper)
def discriminator_loss(y_true, y_pred):
    return 0.5 * keras.losses.mse(y_true, y_pred)


normalization_axis = 3
p_scale = True
p_center = True


def get_norm_layer(norm):
    if norm == 'batch':
        layer = BatchNormalization(momentum=0.9)
    elif norm == 'instance':
        layer = InstanceNormalization(axis=normalization_axis, scale=p_scale, center=p_center)
    elif norm == 'layer':
        # as in tensorflow example, different then default parameters
        layer = LayerNormalization(axis=normalization_axis, scale=p_scale, center=p_center)
    elif norm == 'group_8':
        num_groups = 8
        layer = GroupNormalization(groups=num_groups, axis=normalization_axis)
    elif norm == 'group_16':
        num_groups = 16
        layer = GroupNormalization(groups=num_groups, axis=normalization_axis)
    elif norm == 'group_32':
        num_groups = 32
        layer = GroupNormalization(groups=num_groups, axis=normalization_axis)
    elif norm == 'no_instance':
        layer = custum_layers.InstanceNormalization()
    elif norm == 'no_norm':
        layer = Lambda((lambda x: x))

    return layer

def additive_gaussian_noise_layer(x, std_dev):
    gaussian_noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=std_dev, dtype=x.dtype)
    return x + gaussian_noise


class CycleGAN_with_Gaussian_Noise:
    def __init__(self, norm, noise_std_dev, lambda_cycle=10.0, lambda_id=5.0):
        # Input shape is kept according to the input shape of U-Net model
        self.img_rows = 508
        self.img_cols = 508
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = lambda_cycle  # Cycle-consistency loss
        self.lambda_id = lambda_id  # Identity loss

        self.optimizer = Adam(0.0002)
        self.initializer = keras.initializers.RandomNormal(mean=0, stddev=0.02)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator(norm=norm)
        self.d_B = self.build_discriminator(norm=norm)
        self.d_A.compile(loss=discriminator_loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.d_B.compile(loss=discriminator_loss, optimizer=self.optimizer, metrics=['accuracy'])

        # Calculate output shape of D (PatchGAN)
        # Take the output from the discriminator and its shape for the dimensions of the discriminator patch
        patch = self.d_A.output_shape[1]  # round(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)
        print(self.disc_patch)

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator(norm=norm)
        self.g_BA = self.build_generator(norm=norm)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        # Translate images back to original domain
        fake_B_noisy = additive_gaussian_noise_layer(x=fake_B, std_dev=noise_std_dev)
        reconstr_A = self.g_BA(fake_B_noisy)
        fake_A_noisy = additive_gaussian_noise_layer(x=fake_A, std_dev=noise_std_dev)
        reconstr_B = self.g_AB(fake_A_noisy)

        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id],
                              optimizer=self.optimizer)

    # as per implementation on GitHub in torch, they first do normalization and only then activation
    def build_generator(self, norm):

        def c7s1_k(layer_input, k, norm, padding=(3, 3)):
            g = custum_layers.ReflectionPadding2D(padding=padding)(layer_input)
            g = Conv2D(kernel_size=7, strides=1, filters=k, kernel_initializer=self.initializer)(g)
            g = get_norm_layer(norm)(g)
            g = Activation('relu')(g)

            return g

        def dk(layer_imput, k, norm):
            g = Conv2D(kernel_size=3, strides=2, filters=k, kernel_initializer=self.initializer, padding='same')(layer_imput)
            g = get_norm_layer(norm)(g)
            g = Activation('relu')(g)

            return g

        def rk(layer_input, k, norm):
            g = custum_layers.ReflectionPadding2D(padding=(1, 1))(layer_input)
            g = Conv2D(kernel_size=3, strides=1, filters=k, kernel_initializer=self.initializer)(g)
            g = get_norm_layer(norm)(g)
            g = Activation('relu')(g)

            g = custum_layers.ReflectionPadding2D(padding=(1, 1))(g)
            g = Conv2D(kernel_size=3, strides=1, filters=k)(g)
            g = get_norm_layer(norm)(g)
            g = Add()([g, layer_input])
            # activation is applied after connecting to the input
            g = Activation('relu')(g)
            return g

        def uk(layer_input, k, norm):
            g = Conv2DTranspose(kernel_size=3, filters=k, strides=2, padding='same',
                                kernel_initializer=self.initializer)(layer_input)
            g = get_norm_layer(norm)(g)
            g = Activation('relu')(g)
            return g

        print("Generator : norm " + norm)
        g0 = Input(shape=self.img_shape)

        g = c7s1_k(g0, 32, norm)
        g = dk(g, 64, norm)
        g = dk(g, 128, norm)
        for _ in range(9):
            g = rk(g, 128, norm)
        g = uk(g, 64, norm)
        g = uk(g, 32, norm)

        # The last convolutional layer
        g = custum_layers.ReflectionPadding2D(padding=(3, 3))(g)
        g = Conv2D(kernel_size=7, strides=1, filters=3, kernel_initializer=self.initializer)(g)
        g = Activation('tanh')(g)

        model = Model(g0, g)

        return model

    def build_discriminator(self, norm):

        def ck(layer_input, k, norm):
            g = ZeroPadding2D(padding=(1, 1))(layer_input)
            g = Conv2D(kernel_size=4, strides=2, filters=k, kernel_initializer=self.initializer)(g)
            if norm is not None:
                g = get_norm_layer(norm)(g)
            g = LeakyReLU(0.2)(g)
            return g

        img = Input(shape=self.img_shape)
        d = ck(img, 64, norm=None)
        d = ck(d, 128, norm)
        d = ck(d, 256, norm)

        # To get a 70x70 patch at the output
        d = ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(kernel_size=4, filters=512, strides=1, kernel_initializer=self.initializer)(d)
        d = get_norm_layer(norm)(d)

        d = LeakyReLU(0.2)(d)
        d = ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(kernel_size=4, filters=1, strides=1, kernel_initializer=self.initializer)(d)

        model = Model(img, d)

        return model
