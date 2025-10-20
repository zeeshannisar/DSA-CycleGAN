# -*- coding: utf-8 -*-
#  Odyssee Merveille 09/11/17
"""
background_tissue_detection.py : Generate the background masks for all images of staining _{staincode} in a given directory
"""
import argparse
import numpy
from helper import utils, config_utils_segmentation as config_utils, filepath_utils
import skimage.color
import skimage.morphology
from skimage.filters import threshold_mean
from openslide import OpenSlide
import sys


MIN_OBJ_HOLE_SIZE_AT_LOD_0 = 1700000
PATCH_SIZE = (4096, 4096)


def _calculate_mean(svs, lod):
    mean = 0.0
    k = 0

    imagesize = svs.get_level_dimension(lod)
    imagesize = (imagesize[1], imagesize[0])

    for y in range(0, imagesize[0], PATCH_SIZE[0]):
        for x in range(0, imagesize[1], PATCH_SIZE[1]):
            sizey = min(PATCH_SIZE[0], imagesize[0] - y)
            sizex = min(PATCH_SIZE[1], imagesize[1] - x)

            sys.stdout.write("{:^6}, {:^6}\r".format(x, y))

            patch = svs.read_region((x, y), lod, (sizex, sizey))
            patch = skimage.color.rgb2gray(patch) * 255

            mean = (k * 1.0 / (k + patch.size)) * mean + (patch.size * 1.0 / (k + patch.size)) * numpy.mean(patch)
            k += patch.size

    return mean


def _apply_threshold(svs, lod, threshold):
    imagesize = svs.get_level_dimension(lod)
    imagesize = (imagesize[1], imagesize[0])

    threshImage = numpy.zeros(imagesize, dtype=bool)
    for y in range(0, imagesize[0], PATCH_SIZE[0]):
        for x in range(0, imagesize[1], PATCH_SIZE[1]):
            sizey = min(PATCH_SIZE[0], imagesize[0] - y)
            sizex = min(PATCH_SIZE[1], imagesize[1] - x)
            endy = min(y + PATCH_SIZE[0], imagesize[0])
            endx = min(x + PATCH_SIZE[1], imagesize[1])

            sys.stdout.write("{:^6}, {:^6}\r".format(sizey, sizex))

            patch = svs.read_region((x, y), lod, (sizex, sizey))
            patch = skimage.color.rgb2gray(patch) * 255
            patch = patch.astype(numpy.uint8)

            threshImage[y:endy, x:endx] = patch < threshold

    return threshImage


def generate_tissue_mask_from_disk(imagePath, outputPath, lod):
    """
    generate_tissue_mask: Generates masks of the background of the biopsy sample (excluding the tissue),
    (background = 0, tissue = 1)
	Probably overkill but seems to work...

    :param imagePath: (string) the path of the raw svs image
    :param outputPath: (string) the background folder
    :param lod: (int) the level of detail used
    """


    #  Read the image directly with the right scaleFactor, if it exists in the svs. Otherwise, rescale the image
    svs = utils.open_svs_image_forced(imagePath)
    threshold = _calculate_mean(svs, lod)
    print('Threshold: %s' % threshold)
    threshImage = _apply_threshold(svs, lod, threshold)
    svs.close()

    # keep only the larger structures (the piece of tissues)
    skimage.morphology.remove_small_objects(threshImage, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**lod), in_place=True)
    # Remove the holes inside the piece of tissues
    skimage.morphology.remove_small_holes(threshImage, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**lod), in_place=True)
    # Save the image
    utils.save_binary_image(threshImage, outputPath)


def generate_tissue_mask(imagePath, outputPath, lod):
    """
    generate_tissue_mask: Generates masks of the background of the biopsy sample (excluding the tissue),
    (background = 0, tissue = 1)
	Probably overkill but seems to work...

    :param imagePath: (string) the path of the raw svs image
    :param outputPath: (string) the background folder
    :param lod: (int) the level of detail used
    """
    #  Read the image directly with the right scaleFactor, if it exists in the svs. Otherwise, rescale the image
    image = utils.read_svs_image_forced(imagePath, lod)

    #  transform to grayscale image
    image = skimage.color.rgb2gray(image) * 255
    image = image.astype(numpy.uint8)

    # threshold
    threshold = threshold_mean(image)
    print('Threshold: %s' % threshold)
    threshImage = (image < threshold)

    # keep only the larger structures (the piece of tissues)
    smallHoles = skimage.measure.label(threshImage)
    largePieces = skimage.morphology.remove_small_objects(smallHoles, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**lod))

    # Remove the holes inside the piece of tissues
    largePieces = (largePieces == 0).astype(int)
    smallHoles = skimage.measure.label(largePieces)
    backgroundMask = skimage.morphology.remove_small_objects(smallHoles, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**lod))

    tissueMask = (backgroundMask == 0).astype(numpy.bool_)

    # Save the image
    utils.save_binary_image(tissueMask, outputPath)


def generate_background_mask_from_disk(imagePath, outputPath, lod):
    """
    generate_background_mask_iteratively: Generates masks of the background of the tissue sample (excluding the tissue),
    (background = 0, tissue = 1)
	Probably overkill but seems to work...

    :param imagePath: (string) the path of the raw svs image
    :param outputPath: (string) the background folder
    :param lod: (int) the level of detail used
    """

    svs = utils.open_svs_image_forced(imagePath)
    threshold = _calculate_mean(svs, lod)
    print('Threshold: %s' % threshold)
    threshImage = _apply_threshold(svs, lod, threshold)
    svs.close()

    # keep only the larger structures (the piece of tissues)
    skimage.morphology.remove_small_objects(threshImage, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**lod), in_place=True)
    # Remove the holes inside the piece of tissues
    skimage.morphology.remove_small_holes(threshImage, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**lod), in_place=True)
    # Save the image
    utils.save_binary_image(numpy.logical_not(threshImage), outputPath)


def generate_background_mask(imagePath, outputPath, lod):
    """
    generate_background_mask: Generates masks of the background of the biopsy sample (excluding the tissue),
    (background = 0, tissue = 1)
	Probably overkill but seems to work...

    :param imagePath: (string) the path of the raw svs image
    :param outputPath: (string) the background folder
    :param lod: (int) the level of detail used
    """
    #  Read the image directly with the right scaleFactor, if it exists in the svs. Otherwise, rescale the image
    image = utils.read_svs_image_forced(imagePath, lod)

    #  transform to grayscale image
    image = skimage.color.rgb2gray(image) * 255
    image = image.astype(numpy.uint8)

    # threshold
    threshold = threshold_mean(image)
    print('Threshold: %s' % threshold)
    threshImage = (image < threshold)

    # keep only the larger structures (the piece of tissues)
    smallHoles = skimage.measure.label(threshImage)
    largePieces = skimage.morphology.remove_small_objects(smallHoles, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**lod))

    # Remove the holes inside the piece of tissues
    largePieces = (largePieces == 0).astype(int)
    smallHoles = skimage.measure.label(largePieces)
    backgroundMask = skimage.morphology.remove_small_objects(smallHoles, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**lod))

    # Save the image
    utils.save_binary_image(backgroundMask.astype(numpy.bool_), outputPath)


def generate_tissue_and_background_masks_from_disk(imagePath, tissueOutputPath, backgroundOutputPath, lod):
    """
    generate_background_mask_iteratively: Generates masks of the background of the tissue sample (excluding the tissue),
    (background = 0, tissue = 1)
    Probably overkill but seems to work...

    :param imagePath: (string) the path of the raw svs image
    :param outputPath: (string) the background folder
    :param lod: (int) the level of detail used
    """

    svs = utils.open_svs_image_forced(imagePath)
    if lod < 2:
        lodimagesize = svs.get_level_dimension(lod)
        processing_lod = 2
    else:
        processing_lod = lod
    threshold = _calculate_mean(svs, processing_lod)
    print('Threshold: %s' % threshold)
    threshImage = _apply_threshold(svs, processing_lod, threshold)
    svs.close()

    # keep only the larger structures (the piece of tissues)
    skimage.morphology.remove_small_objects(threshImage, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**processing_lod), in_place=True)
    # Remove the holes inside the piece of tissues
    skimage.morphology.remove_small_holes(threshImage, MIN_OBJ_HOLE_SIZE_AT_LOD_0/(4**processing_lod), in_place=True)
    # Save the image

    if processing_lod > lod:
        lodimagesize = (lodimagesize[1], lodimagesize[0])
        from skimage import transform, img_as_bool
        threshImage = img_as_bool(transform.resize(threshImage, lodimagesize))
        # ind_x = numpy.zeros(lodimagesize[0])
        # ind_y = numpy.zeros(lodimagesize[1])
        # ind_x[0::2] = range(threshImage.shape[0])
        # ind_x[1::2] = range(threshImage.shape[0])
        # ind_y[0::2] = range(threshImage.shape[1])
        # ind_y[1::2] = range(threshImage.shape[1])
        # threshImage = threshImage[ind_y, :]
        # threshImage = threshImage[:, ind_x]

    utils.save_binary_image(threshImage.astype(bool), tissueOutputPath)
    utils.save_binary_image(numpy.logical_not(threshImage.astype(bool)), backgroundOutputPath)

# histo
# ~ fig ,ax = plt.subplots()
# ~ ax.hist(image.ravel(), bins=256)
# ~ ax.set_title('Histogram')
# ~ ax.axvline(threshold, color='r')
# ~ plt.show(block = False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    filePath = filepath_utils.FilepathGenerator(config)

    for imPath, imageName in filePath.get_images(config['extraction.imagepath'], config['general.staincode']):
        print('Image: %s' % imageName)

        tissueOutputPath = filePath.generate_maskpath(imageName, 'tissue', config['general.lod'])
        backgroundOutputPath = filePath.generate_maskpath(imageName, 'background', config['general.lod'])

        generate_tissue_and_background_masks_from_disk(imPath, tissueOutputPath, backgroundOutputPath, config['general.lod'])

        #generate_tissue_mask_from_disk(imPath, tissueOutputPath, config['general.lod'])
        #generate_tissue_mask(imPath, tissueOutputPath, config['general.lod'])

        #generate_background_mask(imPath, backgroundOutputPath, config['general.lod'])
        #generate_background_mask_from_disk(imPath, backgroundOutputPath, config['general.lod'])

