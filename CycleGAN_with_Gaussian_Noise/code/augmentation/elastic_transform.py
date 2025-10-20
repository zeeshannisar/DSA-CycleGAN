# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

import numpy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import glob
import os
import time
from utils import image_utils
from tensorflow.keras.utils import to_categorical


def generate_from_directory(classname, numberofsamples, datapath, outputpath, sigma=10, alpha=100, changefilename=True):

    os.makedirs(os.path.join(outputpath, 'images', classname), exist_ok=True)
    os.makedirs(os.path.join(outputpath, 'gts', classname), exist_ok=True)

    filenames = glob.glob(os.path.join(datapath, 'images', classname, "*.png"))

    # Get random indexes for the number of images to generate
    idx = numpy.random.randint(len(filenames), size=numberofsamples)

    for c, ind in enumerate(idx):
        if changefilename:
            save_filename = os.path.splitext(os.path.basename(filenames[ind]))[0] + '_' + str(c) + '.png'
        else:
            save_filename = os.path.basename(filenames[ind])

        image = image_utils.read_image(filenames[ind])
        maskfilename = os.path.join(datapath, 'gts', classname, os.path.splitext(os.path.basename(filenames[ind]))[0] + '.png')
        mask = image_utils.read_image(maskfilename)

        img_result, msk_result = transform(image, mask, sigma=sigma, alpha=alpha)

        image_utils.save_image(img_result.astype('uint8'), os.path.join(outputpath, 'images', classname, save_filename))

        image_utils.save_image(msk_result.astype('uint8'), os.path.join(outputpath, 'gts', classname, save_filename))


def transform(image, mask, sigma=10, alpha=100, seed=None):

    if seed is None:
        seed = int(time.time())

    if len(mask.shape) == 2:
        mask = mask[:, :, numpy.newaxis]

    if mask.shape[2] > 1:
        raise Exception('Mask must only have one channel')

    nb_classes = int(numpy.max(mask))+1

    mask = to_categorical(mask, num_classes=nb_classes)

    img_result = __elastic_transform(image, alpha=alpha, sigma=sigma, random_state=seed)
    msk_result = __elastic_transform(mask, alpha=alpha, sigma=sigma, random_state=seed)

    # img_result = __elastic_transform2(image, kernel_dim=21, sigma=6, alpha=30, negated=False)
    # msk_result = __elastic_transform2(mask, kernel_dim=21, sigma=6, alpha=30, negated=False)

    msk_result = numpy.argmax(msk_result, axis=-1)

    return img_result, msk_result


'''
Version 1
'''
def __elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    random_state = numpy.random.RandomState(random_state)

    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]), indexing='ij')
    indices = numpy.reshape(x+dx, (-1, 1)), numpy.reshape(y+dy, (-1, 1))

    if len(image.shape) > 2:
        # Image is RGB
        distorted = numpy.empty(image.shape, dtype=image.dtype)
        for i in range(image.shape[2]):
            distorted[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode='reflect').reshape(shape)
    else:
        # Image is greyscale
        distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    return distorted

def __create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma
    :param dim: integer denoting a side (1-d) of gaussian kernel
    :param sigma: floating point indicating the standard deviation
    :returns: a numpy 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = numpy.zeros((dim, dim), dtype=numpy.float16)

    # calculate the center point
    center = dim/2

    # calculate the variance
    variance = sigma ** 2

    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2*variance

            kernel[x,y] = coeff * numpy.exp(-1. * numerator/denom)

    return kernel/sum(sum(kernel))


'''
Version 2
'''
def __elastic_transform2(image, kernel_dim=21, sigma=6, alpha=30, negated=True):
    """
    This method performs elastic transformations on an image by convolving
    with a gaussian kernel.
    :param image: a numpy nd array
    :kernel_dim: dimension(1-D) of the gaussian kernel
    :param sigma: standard deviation of the kernel
    :param alpha: a multiplicative factor for image after convolution
    :param negated: a flag indicating whether the image is negated or not
    :returns: a nd array transformed image
    """
    # check if the image is a negated one
    if not negated:
        image = 255-image

    # check if kernel dimension is odd
    if kernel_dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # create an empty image
    result = numpy.zeros(image.shape)

    # create random displacement fields
    displacement_field_x = numpy.array([[env.numpy_rand.random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha
    displacement_field_y = numpy.array([[env.numpy_rand.random_integers(-1, 1) for x in xrange(image.shape[0])] \
                            for y in xrange(image.shape[1])]) * alpha

    # create the gaussian kernel
    kernel = __create_2d_gaussian(kernel_dim, sigma)

    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel)
    displacement_field_y = convolve2d(displacement_field_y, kernel)

    # make the distortrd image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields

    for row in xrange(image.shape[1]):
        for col in xrange(image.shape[0]):
            low_ii = row + int(math.floor(displacement_field_x[row, col]))
            high_ii = row + int(math.ceil(displacement_field_x[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
               or high_jj >= image.shape[0] - 1:
                continue

            res = image[low_ii, low_jj]/4 + image[low_ii, high_jj]/4 + \
                    image[high_ii, low_jj]/4 + image[high_ii, high_jj]/4

            result[row, col] = res

    return result
