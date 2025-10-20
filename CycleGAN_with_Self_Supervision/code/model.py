import tensorflow as tf
import tensorflow_addons as tfa
import custom_layers


def get_norm_layer(norm, momentum=0.9, axis=3, scale=True, center=True, groups=8):
    if norm == 'batch':
        return tf.keras.layers.BatchNormalization(momentum=momentum)
    elif norm == 'instance':
        return tfa.layers.InstanceNormalization(axis=axis, scale=scale, center=center)
    elif norm == 'layer':
        return tf.keras.Layer.Normalization(axis=axis, scale=scale, center=center)
    elif norm == 'group':
        return tfa.layers.GroupNormalization(groups=groups, axis=axis)
    elif norm == 'none':
        return lambda: lambda x: x


class CycleGAN:
    def __init__(self, norm="instance", extra_channel=False, extra_channel_mode=None):
        # Input shape is kept according to the input shape of U-Net model
        self.img_rows = 508
        self.img_cols = 508
        self.channels = 3
        self.extra_channel = extra_channel
        self.extra_channel_mode = extra_channel_mode

        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_adv = 1.0  # Adversarial loss
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 0.5 * self.lambda_cycle  # Identity loss (half of the cycle-consistency)

        # Optimizer
        self.G_optimizer = tf.keras.optimizers.Adam(0.0002)
        self.D_optimizer = tf.keras.optimizers.Adam(0.0002)

        # As suggested in paper Weights are initialized from a Gaussian distribution with mean 0 and std-dev 0.02.
        self.initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)

        # Build and compile the discriminators
        self.D_A = self.build_discriminator(norm=norm)
        self.D_B = self.build_discriminator(norm=norm)

        # Build the generators
        self.G_AB = self.build_generator(norm=norm)
        self.G_BA = self.build_generator(norm=norm)

    # First normalization and then activation (at their original implementation)
    def build_generator(self, norm, name=None):

        def c7s1_k(layer_input, k, normalisation, padding=(3, 3)):
            g = custom_layers.ReflectionPadding2D(padding=padding)(layer_input)
            g = tf.keras.layers.Conv2D(kernel_size=7, strides=1, filters=k, kernel_initializer=self.initializer)(g)
            g = get_norm_layer(normalisation)(g)
            g = tf.keras.layers.Activation('relu')(g)
            return g

        def dk(layer_input, k, normalisation, padding='same'):
            g = tf.keras.layers.Conv2D(kernel_size=3, strides=2, filters=k, kernel_initializer=self.initializer,
                                       padding=padding)(layer_input)
            g = get_norm_layer(normalisation)(g)
            g = tf.keras.layers.Activation('relu')(g)
            return g

        def rk(layer_input, k, normalisation, padding=(1, 1)):
            g = custom_layers.ReflectionPadding2D(padding=padding)(layer_input)
            g = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=k, kernel_initializer=self.initializer)(g)
            g = get_norm_layer(normalisation)(g)
            g = tf.keras.layers.Activation('relu')(g)

            g = custom_layers.ReflectionPadding2D(padding=padding)(g)
            g = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=k)(g)
            g = get_norm_layer(normalisation)(g)
            g = tf.keras.layers.Add()([g, layer_input])
            # As in the paper, activation is applied after adding
            g = tf.keras.layers.Activation('relu')(g)
            return g

        def uk(layer_input, k, normalisation, padding='same'):
            g = tf.keras.layers.Conv2DTranspose(kernel_size=3, filters=k, strides=2, padding=padding,
                                                kernel_initializer=self.initializer)(layer_input)
            g = get_norm_layer(normalisation)(g)
            g = tf.keras.layers.Activation('relu')(g)
            return g

        if self.extra_channel:
            if self.extra_channel_mode == 'grayscale':
                img_shape = (self.img_rows, self.img_cols, self.channels + 1)
            elif self.extra_channel_mode == 'rgb':
                img_shape = (self.img_rows, self.img_cols, self.channels + 3)
            else:
                raise ValueError("ExtraChannelMode to save the Noisy information should be one of ['grayscale', 'rgb']")
        else:
            img_shape = (self.img_rows, self.img_cols, self.channels)

        g0 = tf.keras.layers.Input(shape=img_shape)

        g = c7s1_k(g0, 32, norm)
        g = dk(g, 64, norm)
        g = dk(g, 128, norm)
        for _ in range(9):
            g = rk(g, 128, norm)
        g = uk(g, 64, norm)
        g = uk(g, 32, norm)

        # The last convolutional layer
        g = custom_layers.ReflectionPadding2D(padding=(3, 3))(g)

        g = tf.keras.layers.Conv2D(kernel_size=7, strides=1, filters=img_shape[-1],
                                   kernel_initializer=self.initializer)(g)
        g = tf.keras.layers.Activation('tanh')(g)

        model = tf.keras.models.Model(inputs=g0, outputs=g)
        return model

    def build_discriminator(self, norm):

        def ck(layer_input, k, normalisation):
            # To get a 70x70 patch at the output: added zero padding
            d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(layer_input)
            d = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=k, kernel_initializer=self.initializer)(d)
            if normalisation is not None:
                d = get_norm_layer(normalisation)(d)
            d = tf.keras.layers.LeakyReLU(0.2)(d)
            return d

        img_shape = (self.img_rows, self.img_cols, self.channels)
        d0 = tf.keras.layers.Input(shape=img_shape)
        d = ck(d0, 64, normalisation=None)
        d = ck(d, 128, norm)
        d = ck(d, 256, norm)

        # To get a 70x70 patch at the output
        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = tf.keras.layers.Conv2D(kernel_size=4, filters=512, strides=1, kernel_initializer=self.initializer)(d)
        d = get_norm_layer(norm)(d)

        d = tf.keras.layers.LeakyReLU(0.2)(d)
        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = tf.keras.layers.Conv2D(kernel_size=4, filters=1, strides=1, kernel_initializer=self.initializer)(d)

        model = tf.keras.models.Model(inputs=d0, outputs=d)
        return model
