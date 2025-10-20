import os
import random

import tensorflow as tf
import tensorflow.keras.backend as K

from glob import glob
from dataloader.pillow_API import load_img
from helper.utils import getvalidinputsize


def standardise(image):
    return (image - image.min()) / ((image.max() - image.min()) + K.epsilon())


def preprocess(image):
    return (image / 127.5) - 1


class TFDataLoader:
    def __init__(self, dataDir, mode='train', imageSize=(512, 512, 3), batchSize=1):
        """
        - Using Command Line arguments
        """
        self.dataDir = dataDir
        self.mode = mode
        self.imageSize = getvalidinputsize(imageSize, 5, 3)
        self.batchSize = batchSize
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def GetPNGImageFiles(self, num_examples_mode=False):
        stain_code = self.dataDir.rsplit("/", 3)[1]
        filenames = glob(os.path.join(self.dataDir, self.mode, "images", "images", "*.png"))[:]
        random.shuffle(filenames)

        if num_examples_mode:
            return len(filenames)
        else:
            print(f"Found {len(filenames)} images belonging to stain: '{stain_code}'.")
            return filenames

    def PILImagesRead(self, paths):
        images = load_img(paths, grayscale=False, target_size=(self.imageSize[0], self.imageSize[1]), data_format=None)
        return preprocess(images)

    def ParseImages(self, paths):
        return tf.numpy_function(func=self.PILImagesRead, inp=[paths], Tout=tf.float32)

    def PerformanceConfiguration(self, ds):
        buffer_size = 500 if self.imageSize[0] <= 128 else 100
        ds = ds.shuffle(self.batchSize * buffer_size)
        ds = ds.batch(self.batchSize, drop_remainder=False)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def LoadDataset(self):
        filenames_ds = tf.data.Dataset.from_tensor_slices(self.GetPNGImageFiles(num_examples_mode=False))
        ds = filenames_ds.map(self.ParseImages, num_parallel_calls=self.AUTOTUNE)
        ds = self.PerformanceConfiguration(ds)
        return ds