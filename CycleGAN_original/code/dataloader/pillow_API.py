import math
import numpy as np
from tensorflow.keras import backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def standardise(image):
    return (image - image.min()) / ((image.max() - image.min()) + K.epsilon())

def preprocess(image):
    return (image / 127.5) - 1

def getvalidinputsize(inp_shape, depth=5, k_size=3, data_format='channels_last'):
    convolutions_per_layer = 2

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)

    def calculate(dim_size):
        # Calculate what the last feature map size would be with this patch size
        for _ in range(depth - 1):
            dim_size = (dim_size - ((k_size - 1) * convolutions_per_layer)) / 2
        dim_size -= (k_size - 1) * 2

        # Minimum possible size of last feature map
        if dim_size < 4:
            dim_size = 4

        # Round to the next smallest even number
        dim_size = math.floor(dim_size / 2.) * 2
        # Calculate the original patch size to give this (valid) feature map size
        for _ in range(depth - 1):
            dim_size = (dim_size + (k_size - 1) * convolutions_per_layer) * 2
        dim_size += (k_size - 1) * 2

        return int(dim_size)

    if data_format == 'channels_last':
        spatial_dims = range(len(inp_shape))[:-1]
    elif data_format == 'channels_first':
        spatial_dims = range(len(inp_shape))[1:]

    inp_shape = list(inp_shape)
    for d in spatial_dims:
        inp_shape[d] = calculate(inp_shape[d])

    return tuple(inp_shape)

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def load_img(path, grayscale=False, target_size=None, data_format=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)

    img = pil_image.open(path)

    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')

    img = img_to_array(img, data_format=data_format)

    if target_size is not None:

        if data_format == 'channels_first':
            img_size = img.shape[1:]
        elif data_format:
            img_size = img.shape[:-1]

        if img_size[0] < target_size[0] or img_size[1] < target_size[1]:
            raise ValueError('Invalid cropped image size (%s). Image is %d x %d and target size is %d x %d.' % (path, img_size[0], img_size[1], target_size[0], target_size[1]))

        if (img_size[0] - target_size[0]) % 2 != 0:
            raise ValueError('Invalid cropped image size. There should be an even difference between the image and target heights')

        if (img_size[1] - target_size[1]) % 2 != 0:
            raise ValueError('Invalid cropped image size. There should be an even difference between the image and target widths')

        if img_size != target_size:
            diffs = np.subtract(img_size, target_size)
            diffs //= 2

            img = img[diffs[0]:img_size[0]-diffs[0], diffs[1]:img_size[1]-diffs[1]]

    return img




