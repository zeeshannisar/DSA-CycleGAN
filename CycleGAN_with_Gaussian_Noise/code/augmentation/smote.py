# -*- coding: utf-8 -*-
#  Thomas Lampert 23/11/17

import glob
#Jelica
import sys
import six
sys.modules['sklearn.externals.six'] = six
from imblearn.over_sampling import SMOTE
import os
import numpy
import time
from sklearn.neighbors import NearestNeighbors
import uuid
from utils import image_utils


def generate_from_directory(classname, numberofsamples, datapath, outputpath, nneighbours):

    os.makedirs(os.path.join(outputpath, 'images', classname), exist_ok=True)
    os.makedirs(os.path.join(outputpath, 'gts', classname), exist_ok=True)

    # Load images
    pos_images = []
    pos_masks = []
    pos_labels = []
    pos_filenames = []
    for file in glob.glob(os.path.join(datapath, 'images', classname, "*.png")):
        x = image_utils.read_image(file)
        patch_shape = x.shape
        pos_images.append(x.reshape((1,) + x.shape))

        maskfilename = os.path.join(datapath, 'gts', classname, os.path.splitext(os.path.basename(file))[0] + '.png')
        x = image_utils.read_image(maskfilename)
        x[x > 0] = 1
        pos_masks.append(x.reshape((1,) + x.shape))

        pos_labels.append(1)
        pos_filenames.append(file)

    pos_images = numpy.array([image.reshape(1, -1).tolist()[0] for image in pos_images])
    pos_masks = numpy.array([mask.reshape(1, -1).tolist()[0] for mask in pos_masks])

    # Initialise SMOTE generator
    seed = int(time.time())
    sm_image = SMOTE(k_neighbors=nneighbours, random_state=seed)
    sm_mask = SMOTE(k_neighbors=nneighbours, random_state=seed)

    # Get random indexes for the number of images to generate
    idx = numpy.random.permutation(pos_images.shape[0])
    if numberofsamples < idx.size:
        idx = idx[:numberofsamples]

    nns = NearestNeighbors(n_neighbors=nneighbours + 1).fit(pos_images).kneighbors(pos_images[idx, :], return_distance=False)
    tmp_labels = [1] * (nneighbours + 1) + [0] * (nneighbours + 2)

    for i in range(numberofsamples):

        ind = numpy.random.randint(idx.size)

        # Select nearest neighbours for the current sample
        nn_images = pos_images[nns[ind, ], ]
        nn_masks = pos_masks[nns[ind, ], ]

        # Use dummy variables for the negative class (increasing count by 1)
        neg_images = numpy.concatenate((nn_images, nn_images[:1, ]), 0)
        neg_masks = numpy.concatenate((nn_masks, nn_masks[:1, ]), 0)

        tmp_images = numpy.concatenate((nn_images, neg_images), 0)
        tmp_masks = numpy.concatenate((nn_masks, neg_masks), 0)

        X_image_res, y_image_res = sm_image.fit_sample(tmp_images, tmp_labels)
        X_mask_res, y_mask_res = sm_mask.fit_sample(tmp_masks, tmp_labels)

        #X_mask_res[X_mask_res >= nneighbours / (2*nneighbours)] = 1
        #X_mask_res[X_mask_res < nneighbours / (2*nneighbours)] = 0

        X_mask_res = numpy.rint(X_mask_res)

        save_filename = 'smote_' + str(uuid.uuid4()) + '.png'

        image_utils.save_image(X_image_res[-1,].reshape(patch_shape).astype('uint8'),
                               os.path.join(outputpath, 'images', classname, save_filename))
        image_utils.save_image(X_mask_res[-1,].reshape(patch_shape[:2]).astype('uint8'),
                               os.path.join(outputpath, 'gts', classname, save_filename))


def generate_sample(images, masks, idx, nneighbours, seed=None):

    patch_shape = images.shape

    images = numpy.array([image.reshape(1, -1).tolist()[0] for image in images])
    masks = numpy.array([mask.reshape(1, -1).tolist()[0] for mask in masks])

    # Initialise SMOTE generator
    if seed is None:
        seed = int(time.time())

    sm_image = SMOTE(k_neighbors=nneighbours, random_state=seed)
    sm_mask = SMOTE(k_neighbors=nneighbours, random_state=seed)

    # Get random indexes for the number of images to generate
    numberofsamples = 1

    sample = images[idx, :][numpy.newaxis, ...]
    nns = NearestNeighbors(n_neighbors=nneighbours + 1).fit(images).kneighbors(sample, return_distance=False)
    tmp_labels = [1] * (nneighbours + 1) + [0] * (nneighbours + 2)

    for i in range(numberofsamples):

        ind = numpy.random.randint(idx.size)

        # Select nearest neighbours for the current sample
        nn_images = images[nns[ind, ], ]
        nn_masks = masks[nns[ind, ], ]

        # Use dummy variables for the negative class (increasing count by 1)
        neg_images = numpy.concatenate((nn_images, nn_images[:1, ]), 0)
        neg_masks = numpy.concatenate((nn_masks, nn_masks[:1, ]), 0)

        tmp_images = numpy.concatenate((nn_images, neg_images), 0)
        tmp_masks = numpy.concatenate((nn_masks, neg_masks), 0)

        X_image_res, y_image_res = sm_image.fit_sample(tmp_images, tmp_labels)
        X_mask_res, y_mask_res = sm_mask.fit_sample(tmp_masks, tmp_labels)

        #X_mask_res[X_mask_res >= nneighbours / (2*nneighbours)] = 1
        #X_mask_res[X_mask_res < nneighbours / (2*nneighbours)] = 0

        X_mask_res = numpy.rint(X_mask_res)

        X_image_res = X_image_res[-1, ].reshape(patch_shape).astype('uint8')
        X_mask_res = X_mask_res[-1, ].reshape(patch_shape[:2]).astype('uint8')

        return X_image_res, X_mask_res
