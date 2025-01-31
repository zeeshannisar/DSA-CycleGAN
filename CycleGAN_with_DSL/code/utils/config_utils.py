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

    configdict['config.filename'] = config_file

    # General
    configdict['general.base_dir']          = config.get('general', 'base_dir')
    configdict['general.subset']            = config.get('general', 'subset')
    configdict['general.source_stain']      = config.get('general', 'source_stain')
    configdict['general.target_stain']      = config.get('general', 'target_stain')


    # Training
    configdict['training.batchSize']        = config.getint('training', 'batchSize')
    configdict['training.normalization']    = config.get('training', 'normalization').lower()
    configdict['training.epochs']           = config.getint('training', 'epochs')
    configdict['training.sampleInterval']   = config.getint('training', 'sampleInterval')
    configdict['training.outdir']           = config.get('training', 'outdir')

    return configdict
