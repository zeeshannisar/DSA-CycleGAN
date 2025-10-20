from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, Reshape, Activation, Conv2DTranspose, Cropping2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import he_normal
import tensorflow.keras.backend as K
import math


def getvalidinputsize(inp_shape, depth=5, k_size=3, data_format='channels_last'):

    convolutions_per_layer = 2

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)

    def calculate(dim_size):
        # Calculate what the last feature map size would be with this patch size
        for _ in range(depth-1):
            dim_size = (dim_size - ((k_size-1) * convolutions_per_layer)) / 2
        dim_size -= (k_size-1)*2
        print('dim_size: {}'.format(dim_size))

        # Minimum possible size of last feature map
        if dim_size < 4:
            dim_size = 4

        # Round to the next smallest even number
        dim_size = math.floor(dim_size / 2.) * 2
        # Calculate the original patch size to give this (valid) feature map size
        for _ in range(depth - 1):
            dim_size = (dim_size + (k_size-1) * convolutions_per_layer) * 2
        dim_size += (k_size-1)*2
        print('dim_size: {}'.format(dim_size))

        return int(dim_size)

    if data_format == 'channels_last':
        spatial_dims = range(len(inp_shape))[:-1]
    elif data_format == 'channels_first':
        spatial_dims = range(len(inp_shape))[1:]

    inp_shape = list(inp_shape)
    print(inp_shape)
    for d in spatial_dims:
        inp_shape[d] = calculate(inp_shape[d])

    return tuple(inp_shape)


def getoutputsize(inp_shape, depth=5, k_size=3, padding='valid', data_format='channels_last'):

    convolutions_per_layer = 2

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)

    def calculate_bridge_end_size(dim_size):
        # Calculate what the last feature map size would be with this patch size
        for _ in range(depth-1):
            dim_size = (dim_size - ((k_size-1) * convolutions_per_layer)) / 2
        dim_size -= (k_size-1)*2

        # Minimum possible size of last feature map
        if dim_size < 4:
            dim_size = 4

        # Round to the next smallest even number
        dim_size = math.floor(dim_size / 2.) * 2

        return dim_size

    def calculate_output_size(dim_size):
        # Calculate what the last feature map size would be with this patch size
        for _ in range(depth-1):
            dim_size = (dim_size * 2) - ((k_size-1) * convolutions_per_layer)

        return dim_size

    if data_format == 'channels_last':
        spatial_dims = range(len(inp_shape))[:-1]
    elif data_format == 'channels_first':
        spatial_dims = range(len(inp_shape))[1:]

    inp_shape = list(inp_shape)
    if padding == 'valid':
        otp_shape = [0] * len(inp_shape)
        for d in spatial_dims:
            inp_shape[d] = calculate_bridge_end_size(inp_shape[d])
            otp_shape[d] = calculate_output_size(inp_shape[d])
    else:
        otp_shape = inp_shape

    return tuple(otp_shape)


def getmergeaxis():
    """
        getmergeaxis: get the correct merge axis depending on the backend (TensorFlow or Theano) used by Keras. It is
        used in the concatenation of features maps

    :return:  (int) the merge axis
    """
    # Feature maps are concatenated along last axis (for tf backend, 0 for theano)
    if K.backend() == 'tensorflow':
        merge_axis = -1
    elif K.backend() == 'theano':
        merge_axis = 0
    else:
        raise Exception('Merge axis for backend %s not defined' % K.backend())

    return merge_axis


def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw / 2), int(cw / 2) + 1
        else:
            cw1, cw2 = int(cw / 2), int(cw / 2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch / 2), int(ch / 2) + 1
        else:
            ch1, ch2 = int(ch / 2), int(ch / 2)

        return (ch1, ch2), (cw1, cw2)


def Conv2DLayer(input, filters, kernel_initialiser, kernel_size, padding, batchnormalisation):
    use_bias = not batchnormalisation

    output = Conv2D(filters=filters, kernel_initializer=kernel_initialiser,
                    kernel_size=kernel_size, padding=padding, use_bias=use_bias)(input)
    if batchnormalisation == 'before':
        output = BatchNormalization()(output)
    output = Activation('relu')(output)
    if batchnormalisation == 'after':
        output = BatchNormalization()(output)

    return output


def build_UNet(inp_shape, nb_classes, depth=5, filter_factor_offset=0, initialiser='glorot_uniform', padding='valid', modifiedarch=False, batchnormalisation='before', k_size=3, dropout=False, learnupscale=False):
    """

    build_UNet: build the U-Net model

    Based on:
        Olaf Ronneberger, Philipp Fischer, Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

    :param inp_shape: (tuple) the dimension of the inputs given to the network
            for example if the tuple is (x,y,z) the patches given will have 3 dimensions of shape x * y * z
    :param nb_classes: (int) the number of classes
    :param depth: (int) the number of layers both in the contraction and expansion paths of the U-Net. The whole network
    therefore has a size: 2 * depth
    :param filter_factor_offset: (int) the factor by which to reduce the number of filters used in each convolution (relative to
    the published number)
    :param initialiser: (string) the method used to generate the random initialisation values (default = glorot_uniform)
    :param modifiedarch: (boolean) if True, remove the second convolution layer between the contraction and expansion paths
    :param batchnormalisation: (boolean) enable or disable batch normalisation
    :param k_size:(int) the size of the convolution kernels
    :return:(Model) the U-Net generated
    """

    # get the current merge_axis
    merge_axis = getmergeaxis()

    if padding == 'valid':
        inp_shape = getvalidinputsize(inp_shape, depth, k_size)
    otp_shape = getoutputsize(inp_shape, depth, k_size, padding)

    data = Input(shape=inp_shape, dtype=K.floatx())

    base_filter_power = 6

    ###
    # Contraction path
    ###
    conv_down = [None] * depth
    for i in range(0, depth - 1):
        filters = 2 ** (base_filter_power + i - filter_factor_offset)
        if i == 0:
            conv_down[i] = Conv2DLayer(data, filters, initialiser, k_size, padding, batchnormalisation)
        else:
            conv_down[i] = Conv2DLayer(pool, filters, initialiser, k_size, padding, batchnormalisation)

        conv_down[i] = Conv2DLayer(conv_down[i], filters, initialiser, k_size, padding, batchnormalisation)

        if dropout and i == depth - 2:
            conv_down[i] = Dropout(0.5)(conv_down[i])

        pool = MaxPooling2D(pool_size=(2, 2))(conv_down[i])

    ###
    # Bridge
    ###
    filters = 2 ** (base_filter_power + (depth - 1) - filter_factor_offset)

    conv = Conv2DLayer(pool, filters, initialiser, k_size, padding, batchnormalisation)
    if not modifiedarch:
        conv = Conv2DLayer(conv, filters, initialiser, k_size, padding, batchnormalisation)

    if dropout:
        conv = Dropout(0.5)(conv)

    ###
    # Expansion path
    ###
    base_crop_size = ((k_size - 1) * 2) * 2 # kernel_size -1 reduced on each convolution, two convolutions, 2 x 2 maxpooling
    curr_crop_size = base_crop_size
    for i in range(depth - 2, -1, -1):
        filters = 2 ** (base_filter_power + i - filter_factor_offset)

        if learnupscale:
            up = Conv2DTranspose(filters=filters, kernel_initializer=initialiser,
                                 kernel_size=(2, 2), strides=(2, 2), padding=padding, use_bias=not batchnormalisation)(conv)
            if batchnormalisation == 'before':
                up = BatchNormalization()(up)
            up = Activation('relu')(up)
            if batchnormalisation == 'after':
                up = BatchNormalization()(up)
        else:
            up = UpSampling2D(size=(2, 2))(conv)

        #curr_crop_size = conv_down[i].get_shape()[1].value - up.get_shape()[1].value
        #curr_crop_size = get_crop_shape(up, conv_down[i])

        if padding == 'valid':
            conv_down[i] = Cropping2D(
                cropping=((curr_crop_size // 2, curr_crop_size // 2), (curr_crop_size // 2, curr_crop_size // 2)))(
                conv_down[i])
            curr_crop_size = (2 * curr_crop_size) + (2 * base_crop_size) # 2 x 2 maxpooling mutiplies range of previous crop by 2, plus current crops
        merged = concatenate([conv_down[i], up], axis=merge_axis)

        conv = Conv2DLayer(merged, filters, initialiser, k_size, padding, batchnormalisation)
        conv = Conv2DLayer(conv, filters, initialiser, k_size, padding, batchnormalisation)

    # Classification layer
    out = Conv2D(filters=nb_classes, kernel_initializer=initialiser, kernel_size=1, padding='valid',
                  activation='softmax')(conv)

    model = Model(data, out)

    return model, inp_shape, otp_shape
