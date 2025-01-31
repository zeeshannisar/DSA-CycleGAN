import os
from augmentation.live_augmentation import ImageDataGenerator
from unet.unet_models import getvalidinputsize, getoutputsize

def preprocess_patch(patch):
    patch_p = (patch/127.5) - 1
    return patch_p

def make_generators_from_directory(data_dir, patchSize=512, batch_size=16):
    inp_shape = getvalidinputsize((patchSize, patchSize, 3), 5, 3)
    otp_shape = getoutputsize((patchSize, patchSize, 3), 5, 3, 'valid')

    train_dir = os.path.join(data_dir, "train")

    train_gen = ImageDataGenerator(nb_classes=3, preprocessing_function=preprocess_patch, categoricaltarget=False)
    train_flow = train_gen.flow_from_directory(train_dir, img_target_size=(inp_shape[0],inp_shape[1]),
                                               gt_target_size=(otp_shape[0],otp_shape[1]),
                                               color_mode='rgb',batch_size=batch_size)


    return train_flow
def make_generators_from_directory_whole_image(dir, patchSize=512, batch_size=16):

    inp_shape = (patchSize, patchSize, 3)
    otp_shape = (patchSize, patchSize, 3)

    train_gen = ImageDataGenerator(nb_classes=3,preprocessing_function=preprocess_patch,categoricaltarget=False)
    train_flow = train_gen.flow_from_directory(dir+'/train',img_target_size=(inp_shape[0],inp_shape[1]),
                                               gt_target_size=(otp_shape[0],otp_shape[1]),
                                               color_mode='rgb',batch_size=batch_size)

    valid_gen = ImageDataGenerator(nb_classes=3, preprocessing_function=preprocess_patch, categoricaltarget=False)
    valid_flow = valid_gen.flow_from_directory(dir + '/validation', img_target_size=(inp_shape[0],inp_shape[1]),
                                               gt_target_size=(otp_shape[0],otp_shape[1]),
                                               color_mode='rgb', batch_size=batch_size)


    return  train_flow, valid_flow
