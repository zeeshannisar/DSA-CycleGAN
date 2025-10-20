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
    configdict['dataset.rows']                  = config.getint('dataset', 'rows')
    configdict['dataset.cols']                  = config.getint('dataset', 'cols')
    configdict['dataset.datapath']              = config.get('dataset', 'datapath')

    # Architecture
    configdict['architecture.gf']                   = config.getint('architecture', 'gf')
    configdict['architecture.df']                   = config.getint('architecture', 'df')
    configdict['architecture.c_dim']                = config.getint('architecture', 'c_dim')
    configdict['architecture.num_resnet_blocks']    = config.getint('architecture', 'num_resnet_blocks')

    # Training
    configdict['training.lambda_adv']                            = config.getfloat('training', 'lambda_adv')
    configdict['training.lambda_classification']                = config.getfloat('training', 'lambda_classification')
    configdict['training.lambda_reconstruction']                = config.getfloat('training', 'lambda_reconstruction')
    configdict['training.lambda_gp']                            = config.getfloat('training', 'lambda_gp')
    configdict['training.discriminator_lr']                     = config.getfloat('training', 'discriminator_lr')
    configdict['training.generator_lr']                         = config.getfloat('training', 'generator_lr')
    configdict['training.batch_size']                           = config.getint('training', 'batch_size')
    configdict['training.n_times_discriminator_1_generator']    = config.getint('training', 'n_times_discriminator_1_generator')
    configdict['training.epochs']                               = config.getint('training', 'epochs')


    # Output
    configdict['output.outpath']                        = config.get('output', 'outpath')
    configdict['output.log_step']                       = config.getint('output', 'log_step')
    configdict['output.sample_step']                    = config.getint('output', 'sample_step')

    return configdict