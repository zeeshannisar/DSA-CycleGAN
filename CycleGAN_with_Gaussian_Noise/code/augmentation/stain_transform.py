import numpy
import glob
import os
import time
from utils import image_utils
from stain_normalisation.augmentors import TellezAugmentor


def generate_from_directory(classname, numberofsamples, datapath, outputpath, staincode=None, filePath=None, alpha_range=0., beta_range=0., changefilename=True):

    if not staincode and not filePath:
        raise ValueError('Either staincode or filePath must be set')

    if not staincode:
        calculatestaincode = True

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

        if calculatestaincode:
            staincode = filePath.get_stain(filenames[ind])

        image = image_utils.read_image(filenames[ind])
        maskfilename = os.path.join(datapath, 'gts', classname, os.path.splitext(os.path.basename(filenames[ind]))[0] + '.png')
        mask = image_utils.read_image(maskfilename)

        img_result, msk_result = transform(image, mask, staincode, alpha_range=alpha_range, beta_range=beta_range)

        image_utils.save_image(img_result.astype('uint8'), os.path.join(outputpath, 'images', classname, save_filename))

        image_utils.save_image(msk_result.astype('uint8'), os.path.join(outputpath, 'gts', classname, save_filename))


def transform(image, mask, staincode, alpha_range=0., beta_range=0.):

    dtype = image.dtype

    augmentor = TellezAugmentor('RJ', sigma1=alpha_range, sigma2=beta_range)
    augmentor.fit(image.astype(numpy.uint8).copy(), staincode=staincode, standardize_brightness=False)

    return augmentor.augment(include_background=True).astype(dtype), mask
