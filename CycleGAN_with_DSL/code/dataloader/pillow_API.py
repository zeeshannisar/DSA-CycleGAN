import math
import numpy as np
from tensorflow.keras import backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


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




