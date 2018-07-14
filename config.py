# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataroot = '/home/test2/liaoxingyu/gan/DATA/horse2zebra'
    dataset_mode = 'unaligned'
    phase = 'train'
    resize_or_crop = 'resize_and_crop'
    loadSize = 286  # scale image to this size
    fineSize = 256  # then crop to this size
    which_direction = 'AtoB'
    serial_batches = True
    pool_size = 50
    no_flip = False

    # optimization options
    max_epoch = 200
    batchSize = 1
    beta1 = 0.5
    lr = 2e-4  # initial learning rate adam
    lr_policy = 'lambda'
    lr_decay_iters = 50  # multiply by a gamma every lr_decay_iters iterations
    niter = 100

    # model options
    input_nc = 3  # number of input image channels
    output_nc = 3  # number of output image channels
    ngf = 64  # number of generator filters in first conv layer
    ndf = 64  # number of discriminator filters in first conv layer
    which_model_netD = 'basic'
    which_model_netG = 'resnet_9blocks'
    n_layers_D = 3  # only used if which_model_netD == n_layers
    norm = 'instance'
    no_dropout = True  # no dropout for the generator
    no_lsgan = True
    isTrain = True
    gpu_ids = [0]

    lambda_A = 10.0
    lambda_B = 10.0
    lambda_identity = 0.5

    # miscs
    print_freq = 30
    save_freq = 10
    display_freq = 10
    save_dir = '/home/test2/liaoxingyu/gan/DATA/cyclegan_result'
    workers = 10
    start_epoch = 0

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
