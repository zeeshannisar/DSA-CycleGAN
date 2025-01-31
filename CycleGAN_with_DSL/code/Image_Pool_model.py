import random
import tensorflow as tf
import numpy as np

class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id]
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return tf.stack(return_images, axis=0)


class ImagePoolForDomainShiftLoss:
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enables us to calculate domain shift loss using a history of generated images.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of the image buffer (default: 50)
        """
        self.pool_size = pool_size
        self.images = []  # Stores real source images

    def query(self, images, random_id):
        """
        Stores images and replaces existing ones if the pool is full.
        Parameters:
            images: images.
        Returns:
            A batch of 50 images from each pool when full, else None.
        """
        batch_size = images.shape[0]

        for i in range(batch_size):
            if len(self.images) < self.pool_size:
                # Add images if pool isn't full
                self.images.append(images[i])
            else:
                # Pool is full, replace a random existing image
                self.images[random_id] = images[i]

        # Only return when both pools are full
        if len(self.images) < self.pool_size:
            return None

        return tf.stack(self.images, axis=0)



# if __name__ == '__main__':
#     # Initialize pool with size 5
#     x_pool = ImagePoolForDomainShiftLoss(pool_size=5)
#     y_pool = ImagePoolForDomainShiftLoss(pool_size=5)
#
#     # Create test data (10 numbers)
#     import numpy
#     x = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     y = numpy.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
#
#     # Feed numbers into the pool one by one
#     for i in range(len(x) - 1):
#         random_id = random.randint(0, len(x) - 1)
#         x_batch = numpy.array([x[i]])  # Convert single element into an array
#         y_batch = numpy.array([y[i]])
#
#         x_tmp = x_pool.query(x_batch, random_id)
#         print(f"Step {i + 1}: Added ({x[i]})")
#
#         y_tmp = y_pool.query(y_batch, random_id)
#         print(f"Step {i + 1}: Added ({y[i]})")
#
#         # Only print returned samples when pool is full
#         if x_tmp is not None and y_tmp is not None:
#             print(f"Sampled from pool: {x_tmp.numpy()} - {y_tmp.numpy()}")
#         else:
#             print("Pool is still filling up...")
