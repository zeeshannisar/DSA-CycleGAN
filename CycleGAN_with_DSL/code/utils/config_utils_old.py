"""
config_utils.py: I/O script that simplify the extraction of parameters in a configuration file
"""

import configparser
from utils import image_utils
import os.path
import re


def readconfig(config_file='sysmifta.cfg'):

    if not os.path.isfile(config_file):
        raise ValueError('Config file %s does not exist' % config_file)

    config = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_file)

    configdict = {}

    # General
    configdict['general.base_patch_size']    = config.getint('general', 'base_patch_size')
    configdict['general.lod']                = config.getint('general', 'lod')
    configdict['general.datapath']           = config.get('general', 'datapath')
    configdict['general.staincode']          = config.get('general', 'staincode')
    configdict['general.regexBaseName']      = config.get('general', 'regexBaseName')
    configdict['general.trainPatients']      = config.get('general', 'trainPatients').split(',')
    configdict['general.validationPatients'] = config.get('general', 'validationPatients').split(',')
    configdict['general.testPatients']       = config.get('general', 'testPatients').split(',')

    # Extraction
    configdict['extraction.cytominehost']    = config.get('extraction', 'cytominehost')
    configdict['extraction.projectId']       = config.getint('extraction', 'projectId')
    configdict['extraction.objectIds']       = config.get('extraction', 'objectIds').split(',')
    configdict['extraction.objectLabels']    = config.get('extraction', 'objectLabels').split(',')

    configdict['extraction.extractbasepath'] = config.get('extraction', 'extractbasepath')
    configdict['extraction.imagepath']       = config.get('extraction', 'imagepath')
    configdict['extraction.maskpath']        = config.get('extraction', 'maskpath')
    configdict['extraction.groundtruthpath'] = config.get('extraction', 'groundtruthpath')
    configdict['extraction.patchpath']       = config.get('extraction', 'patchpath')

    # Normalisation
    configdict['normalisation.standardise_patches']     = config.getboolean('normalisation', 'standardise_patches')
    configdict['normalisation.normalise_patches']       = config.getboolean('normalisation', 'normalise_patches')
    configdict['normalisation.normalise_image']         = config.getboolean('normalisation', 'normalise_image')
    configdict['normalisation.normalise_within_tissue'] = config.getboolean('normalisation', 'normalise_within_tissue')

    # Augmentation
    configdict['augmentation.use_augmentation']          = config.getboolean('augmentation', 'use_augmentation')
    configdict['augmentation.live_augmentation']         = config.getboolean('augmentation', 'live_augmentation')
    configdict['augmentation.multiplyexamples']          = config.getboolean('augmentation', 'multiplyexamples')
    configdict['augmentation.multiplyfactor']            = config.getint('augmentation', 'multiplyfactor')
    configdict['augmentation.balanceclasses']            = config.getboolean('augmentation', 'balanceclasses')
    configdict['augmentation.methods']                   = config.get('augmentation', 'methods').split(',')
    configdict['augmentation.patchpath']                 = config.get('augmentation', 'patchpath')
    configdict['augmentation.affine_rotation_range']     = config.getfloat('augmentation', 'affine_rotation_range')
    configdict['augmentation.affine_width_shift_range']  = config.getfloat('augmentation', 'affine_width_shift_range')
    configdict['augmentation.affine_height_shift_range'] = config.getfloat('augmentation', 'affine_height_shift_range')
    configdict['augmentation.affine_rescale']            = config.getfloat('augmentation', 'affine_rescale')
    configdict['augmentation.affine_zoom_range']         = config.getfloat('augmentation', 'affine_zoom_range')
    configdict['augmentation.affine_horizontal_flip']    = config.getboolean('augmentation', 'affine_horizontal_flip')
    configdict['augmentation.affine_vertical_flip']      = config.getboolean('augmentation', 'affine_vertical_flip')
    configdict['augmentation.elastic_sigma']             = config.getfloat('augmentation', 'elastic_sigma')
    configdict['augmentation.elastic_alpha']             = config.getfloat('augmentation', 'elastic_alpha')
    configdict['augmentation.smotenneighbours']          = config.getint('augmentation', 'smotenneighbours')
    configdict['augmentation.stain_alpha_range']         = config.getfloat('augmentation', 'stain_alpha_range')
    configdict['augmentation.stain_beta_range']          = config.getfloat('augmentation', 'stain_beta_range')
    configdict['augmentation.blur_sigma_range']          = [float(i) for i in config.get('augmentation', 'blur_sigma_range').split(',')]
    configdict['augmentation.noise_sigma_range']         = [float(i) for i in config.get('augmentation', 'noise_sigma_range').split(',')]
    configdict['augmentation.bright_factor_range']       = [float(i) for i in config.get('augmentation', 'bright_factor_range').split(',')]
    configdict['augmentation.contrast_factor_range']     = [float(i) for i in config.get('augmentation', 'contrast_factor_range').split(',')]
    configdict['augmentation.colour_factor_range']       = [float(i) for i in config.get('augmentation', 'colour_factor_range').split(',')]

    # Detector
    configdict['detector.inputpath']           = config.get('detector', 'inputpath')
    configdict['detector.modelpath']           = config.get('detector', 'modelpath')
    configdict['detector.outputpath']          = config.get('detector', 'outputpath')
#    configdict['detector.colour_mode']         = config.get('detector', 'colour_mode')
    configdict['detector.lod']                 = config.getint('detector', 'lod')
    configdict['detector.network_depth']       = config.getint('detector', 'network_depth')
    configdict['detector.filter_factor_offset']    = config.getint('detector', 'filter_factor_offset')
    configdict['detector.kernel_size']         = config.getint('detector', 'kernel_size')
    configdict['detector.padding']             = config.get('detector', 'padding')
    configdict['detector.batch_size']          = config.getint('detector', 'batch_size')
    configdict['detector.epochs']              = config.getint('detector', 'epochs')
    configdict['detector.earlyStopping']       = config.getboolean('detector', 'earlyStopping')
    configdict['detector.reducelr']            = config.getboolean('detector', 'reducelr')
    configdict['detector.learn_rate']          = config.getfloat('detector', 'learn_rate')
    configdict['detector.dropout']             = config.getboolean('detector', 'dropout')
    configdict['detector.learnupscale']        = config.getboolean('detector', 'learnupscale')
    configdict['detector.batchnormalisation']  = config.get('detector', 'batchnormalisation')
    configdict['detector.weight_samples']      = config.getboolean('detector', 'weight_samples')
    configdict['detector.weightinit']          = config.get('detector', 'weightinit')
    configdict['detector.modifiedarch']        = config.getboolean('detector', 'modifiedarch')

    # Segmentation
    configdict['segmentation.segmentationpath'] = config.get('segmentation', 'segmentationpath')
    configdict['segmentation.detectionpath']    = config.get('segmentation', 'detectionpath')
    configdict['segmentation.stride']           = config.get('segmentation', 'stride')
    configdict['segmentation.batch_size']       = config.getint('segmentation', 'batch_size')

    # Derived Values
    configdict['detector.patch_size'] = image_utils.getpatchsize(configdict['general.base_patch_size'], configdict['detector.lod'])
    configdict['extraction.patch_size'] = image_utils.getpatchsize(configdict['general.base_patch_size'], configdict['general.lod'])

    if configdict['segmentation.stride'][0] == 'a':
        configdict['segmentation.stride'] = int(configdict['segmentation.stride'][1::])
    elif configdict['segmentation.stride'][0] == 'r':
        if configdict['detector.padding'] == 'same':
            patch_size = configdict['detector.patch_size']
        elif configdict['detector.padding'] == 'valid':
            from unet.unet_models import getvalidinputsize, getoutputsize
            inp_shape = getvalidinputsize((configdict['detector.patch_size'], configdict['detector.patch_size'], 1), configdict['detector.network_depth'], configdict['detector.kernel_size'])
            otp_shape = getoutputsize(inp_shape, configdict['detector.network_depth'], configdict['detector.kernel_size'], configdict['detector.padding'])
            patch_size = otp_shape[0]
        else:
            raise ValueError('invalid detector.padding')
        configdict['segmentation.stride'] = int(float(configdict['segmentation.stride'][1::]) * patch_size)
    else:
        raise ValueError('invalid segmentation.stride (must be preceded by r or a, r = relative and a = absolute)')

    class_merge_dict = {}
    for key in config['classmerges']:
        key = str(key).lower()
        class_merge_dict[key] = config.get('classmerges', key).split(',')
    configdict['extraction.class_merges'] = class_merge_dict

    # Todo: need to check that there are less than 256 classes, label 255 is reserved for error checks

    class_dict = {}
    l = 1
    for key in config['absoluteclassnumbers']:
        key = str(key).lower()
        sample_number = config.getint('absoluteclassnumbers', key)
        if sample_number != 0:
            if key == 'negative':
                label = 0
            else:
                label = l
                l += 1
            # List: classlabel, extraction method, 'absolute', number of samples
            class_dict[key] = [label, config.get('extractionmethods', key).lower(), 'absolute', sample_number]
    for key in config['relativeclassnumbers']:
        key = str(key).lower()
        sample_number = config.get('relativeclassnumbers', key)
        match = re.match(r"([0-9\.]+)([a-z]+)", sample_number, re.I)
        sample_number = float(match.groups()[0])
        target_class = match.groups()[1].lower()
        if sample_number != 0:
            if key == 'negative':
                label = 0
            else:
                label = l
                l += 1
            # List: classlabel, extraction method, 'relative', relative_class_target, number of samples
            class_dict[key] = [label, config.get('extractionmethods', key).lower(), 'relative', target_class, sample_number]
    configdict['extraction.class_definitions'] = class_dict

    #configdict['extraction.positive_counts']

    if not configdict['augmentation.use_augmentation']:
        configdict['augmentation.live_augmentation'] = False

    return configdict