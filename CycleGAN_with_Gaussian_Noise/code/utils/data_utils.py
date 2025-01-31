import numpy
import tensorflow as tf
import tensorflow.keras.backend as K

def standardise_sample(sample):

    # [(image - image.min) / (image.max - image.min)]
    # return (sample - sample.min()) / ((sample.max() - sample.min()) + K.epsilon())
    return (sample - tf.reduce_min(sample)) / ((tf.reduce_max(sample) - tf.reduce_min(sample)) + K.epsilon())


def normalise_sample(sample, mean, stddev):
    # image = [{image - mean(data)} / {std(data)}]
    sample -= mean
    sample /= stddev + K.epsilon()

    return sample
