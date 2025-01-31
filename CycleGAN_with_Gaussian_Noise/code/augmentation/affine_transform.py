# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy
import uuid
import time
import glob
from utils import image_utils
from tensorflow.keras.utils import to_categorical
import warnings


def generate_from_directory(classname, numberofsamples, datapath, outputpath, rotation_range=0,
                            width_shift_range=0, height_shift_range=0, rescale=1.,
                            zoom_range=0, horizontal_flip=False, vertical_flip=False,
                            fill_mode='reflect', cval=0., changefilename=True):

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
        maskfilename = os.path.join(datapath, 'gts', classname,
                                    os.path.splitext(os.path.basename(filenames[ind]))[0] + '.png')
        mask = image_utils.read_image(maskfilename)

        img_result, msk_result = transform(image, mask, rotation_range, width_shift_range, height_shift_range, rescale,
                                           zoom_range, horizontal_flip, vertical_flip, fill_mode, cval)

        image_utils.save_image(img_result.astype('uint8'), os.path.join(outputpath, 'images', classname, save_filename))

        image_utils.save_image(msk_result.astype('uint8'), os.path.join(outputpath, 'gts', classname, save_filename))


def transform(image, mask, rotation_range=0,
              width_shift_range=0, height_shift_range=0, rescale=1.,
              zoom_range=0, horizontal_flip=False, vertical_flip=False,
              fill_mode='reflect', cval=0., seed=None):

    if len(mask.shape) == 2:
        mask = mask[..., numpy.newaxis]

    if len(image.shape) == 2:
        image = image[..., numpy.newaxis]

    if mask.shape[2] > 1:
        raise Exception('Mask must only have one channel')

    if seed is None:
        seed = int(time.time())

    batch_size = 1  # Generate 1 sample per image

    data_gen_args = dict(rotation_range=rotation_range,
                         width_shift_range=width_shift_range,
                         height_shift_range=height_shift_range,
                         rescale=rescale,
                         zoom_range=zoom_range,
                         horizontal_flip=horizontal_flip,
                         vertical_flip=vertical_flip,
                         fill_mode=fill_mode,
                         cval=cval,
                         data_format='channels_last')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    data_flow_args = dict(batch_size=batch_size,
                          shuffle=False,
                          seed=seed)

    nb_classes = int(numpy.max(mask)) + 1

    mask = to_categorical(mask, num_classes=nb_classes)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "NumpyArrayIterator is set to use the")
        image_generator = image_datagen.flow(image[numpy.newaxis, ...], image[numpy.newaxis, ...], **data_flow_args)
        mask_generator = mask_datagen.flow(mask[numpy.newaxis, ...], mask[numpy.newaxis, ...], **data_flow_args)
    train_generator = zip(image_generator, mask_generator)

    for (batch_im, batch_msk) in train_generator:
        return numpy.squeeze(batch_im[0], axis=0), numpy.argmax(numpy.squeeze(batch_msk[0], axis=0), axis=-1)
