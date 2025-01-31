import numpy
import glob
import os
import time
from utils import image_utils
from PIL import Image, ImageEnhance


def generate_from_directory(classname, numberofsamples, datapath, outputpath, bright_factor_range, contrast_factor_range, colour_factor_range, changefilename=True):

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

        img_result, msk_result = transform(image, mask,
                                           bright_factor_range=bright_factor_range,
                                           contrast_factor_range=contrast_factor_range,
                                           colour_factor_range=colour_factor_range)

        image_utils.save_image(img_result.astype('uint8'), os.path.join(outputpath, 'images', classname, save_filename))

        image_utils.save_image(msk_result.astype('uint8'), os.path.join(outputpath, 'gts', classname, save_filename))


def transform(image, mask, bright_factor_range=[1., 1.], contrast_factor_range=[1., 1.], colour_factor_range=[1., 1.]):

    bright_factor=numpy.random.uniform(low=bright_factor_range[0], high=bright_factor_range[1])
    contrast_factor = numpy.random.uniform(low=contrast_factor_range[0], high=contrast_factor_range[1])
    colour_factor = numpy.random.uniform(low=colour_factor_range[0], high=colour_factor_range[1])

    return __enhance_transform(image, bright_factor=bright_factor, contrast_factor=contrast_factor, colour_factor=colour_factor), mask


def __enhance_transform(image, bright_factor=1., contrast_factor=1., colour_factor=1.):

    dtype = image.dtype
    image = Image.fromarray(image.astype(numpy.uint8))

    # Brightness
    image = ImageEnhance.Brightness(image).enhance(bright_factor)

    # Colour
    image = ImageEnhance.Color(image).enhance(colour_factor)

    # Contrast
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    return numpy.asarray(image).astype(dtype)
