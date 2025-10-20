from tensorflow.keras.callbacks import Callback
from utils import image_utils,data_utils
import os
import numpy
import shutil


class CheckpointTrainingPatches(Callback):

    def __init__(self, source_dir, patches_dir, inp_shape, otp_shape, colour_mode, mean=None, stddev=None):
        self.inp_shape = inp_shape
        self.otp_shape = otp_shape
        self.mean = mean
        self.stddev = stddev
        self.source_images_test, self.source_masks_test, self.source_class_number, self.source_filenames = self.__getdataset__(source_dir, colour_mode, True, 8)
        self.output_path = patches_dir

    def __getdataset__(self, input_patch_path, colour_mode, standardise_patches=False, number_of_patches_per_class=8):
        # Read images from directory
        classnames = [x[1] for x in os.walk(os.path.join(input_patch_path, 'images'))][0]
        class_number = len(classnames)
        images_test = []
        masks_test = []
        filenames = []
        for classname in classnames:
            dirlist = os.listdir(os.path.join(input_patch_path, 'images', classname))

            perm = numpy.random.permutation(len(dirlist))
            randomdirlist = [dirlist[i] for i in perm[:number_of_patches_per_class]]

            for filename in randomdirlist:
                image = image_utils.read_image(os.path.join(input_patch_path, 'images', classname, filename)).astype(
                    numpy.float32)
                print(filename)
                image = image_utils.image_colour_convert(image, colour_mode)
                inp_diff = numpy.subtract(list(image.shape[:-1]), list(self.inp_shape[:-1]))
                inp_diff //= 2

                image = image[inp_diff[0]:image.shape[0] - inp_diff[0], inp_diff[1]:image.shape[1] - inp_diff[1], :]
                images_test.append(image)

                mask = image_utils.read_image(os.path.join(input_patch_path, 'gts', classname, filename)).astype(
                    numpy.float32)
                mask = numpy.expand_dims(mask, axis=2)
                otp_diff = numpy.subtract(list(mask.shape[:-1]), list(self.otp_shape[:-1]))
                otp_diff //= 2

                mask = mask[otp_diff[0]:mask.shape[0] - otp_diff[0], otp_diff[1]:mask.shape[1] - otp_diff[1], :]

                masks_test.append(mask)

                filenames.append(filename)

        images_test = numpy.array(images_test)
        masks_test = numpy.array(masks_test)

        if standardise_patches:
            for idx, sample in enumerate(images_test):
                images_test[idx, ] = data_utils.standardise_sample(images_test[idx, ])

        # Normalise data
        if self.mean and self.stddev:
            for idx, sample in enumerate(images_test):
                images_test[idx,] = data_utils.normalise_sample(images_test[idx, ], self.mean, self.stddev)

        return images_test, masks_test, class_number, filenames

    def __predictions__(self, images_test, filenames, class_number, model, output_path, epoch):
        y_pred = model.predict(images_test.astype(model.inputs[0].dtype.name))

        img_output_path = os.path.join(output_path, 'predictions', str(epoch))
        if os.path.exists(img_output_path):
            shutil.rmtree(img_output_path, ignore_errors=True)
        os.makedirs(img_output_path)

        for pred, filename in zip(y_pred, filenames):
            for c in range(class_number):
                image_utils.save_image((pred[..., c]*255).astype(numpy.uint8), os.path.join(img_output_path, os.path.splitext(os.path.basename(filename))[0] + '_' + str(c) + '.png'))

    def on_epoch_end(self, epoch, logs):
        self.__predictions__(self.source_images_test, self.source_filenames, self.source_class_number, self.model, self.output_path, epoch)
