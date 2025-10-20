import numpy
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import image_utils

def standardise_sample(sample):

    return (sample - sample.min()) / ((sample.max() - sample.min()) + K.epsilon())


def standardise_sample_tf(sample):
    sample_min = tf.reduce_min(sample, axis=[0, 1], keepdims=True)  # Min value across height and width
    sample_max = tf.reduce_max(sample, axis=[0, 1], keepdims=True)  # Max value across height and width
    standardized_sample = (sample - sample_min) / (sample_max - sample_min + K.epsilon())
    return standardized_sample


def normalise_sample(sample, mean, stddev):
    sample -= mean
    sample /= stddev + K.epsilon()
    return sample

def save_img(img, img_path):
    image_utils.save_image(img,img_path)

def take_central_path_of_shape(img,shape):
    x = shape[0]
    y = shape[1]
    x_off = (img.shape[0]-x)//2
    y_off = (img.shape[1] - y) // 2
    # print('x='+str(x)+' y='+str(y)+' x_off='+str(x_off)+' y_off='+str(y_off))
    return img[x_off:img.shape[0]-x_off, y_off:img.shape[1]-y_off]
