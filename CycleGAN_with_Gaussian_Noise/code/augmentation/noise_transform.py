import numpy
import glob
import os
import time
from utils import image_utils
from skimage.util import random_noise


def generate_from_directory(classname, numberofsamples, datapath, outputpath, sigma_range=[0., 0.], changefilename=True):

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

        img_result, msk_result = transform(image, mask, sigma_range=sigma_range)

        image_utils.save_image(img_result.astype('uint8'), os.path.join(outputpath, 'images', classname, save_filename))

        image_utils.save_image(msk_result.astype('uint8'), os.path.join(outputpath, 'gts', classname, save_filename))


def transform(image, mask, sigma_range=[0., 0.]):

    sigma = numpy.random.uniform(low=sigma_range[0], high=sigma_range[1])

    return __noise_transform(image, sigma=sigma), mask


def __noise_transform(image, sigma=0.):

    dtype = image.dtype

    img_result = random_noise(image.astype(numpy.uint8), clip=True, mode='gaussian', var=sigma ** 2) * 255

    return img_result.astype(dtype)
