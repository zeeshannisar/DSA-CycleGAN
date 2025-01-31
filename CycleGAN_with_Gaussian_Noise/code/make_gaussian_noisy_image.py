import os
import numpy as np
import tensorflow as tf
from utils import image_utils

stddev_list = [0.0125, 0.025, 0.05, 0.075, 0.1, 0.3, 0.9]
for stddev in stddev_list:
    gaussian_noise = tf.random.normal(shape=(508, 508, 3), mean=0.0, stddev=stddev, dtype=tf.dtypes.float32).numpy()
    gaussian_noise = 0.5 * gaussian_noise + 0.5
    os.makedirs("./gaussian_noisy_images", exist_ok=True)
    image_utils.save_image((gaussian_noise*255.0).astype(np.uint8), savePath=f"./gaussian_noisy_images/stddev_{stddev}.png")
