# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchvision.transforms as T
from PIL import Image


class ImageTransform(object):
    def __init__(self, opt):
        transform_list = []
        if opt.resize_or_crop == 'resize_and_crop':
            osize = [opt.loadSize, opt.loadSize]
            transform_list.append(T.Resize(osize, Image.BICUBIC))
            transform_list.append(T.RandomCrop(opt.fineSize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(T.RandomCrop(opt.fineSize))
        elif opt.resize_or_crop == 'scale_width':
            transform_list.append(T.Lambda(
                lambda img: __scale_width(img, opt.fineSize)
            ))
        elif opt.resize_or_crop == 'scale_width_and_crop':
            transform_list.append(T.Lambda(
                lambda img: __scale_width(img, opt.loadSize)
            ))
            transform_list.append(T.RandomCrop(opt.fineSize))
        elif opt.resize_or_crop == 'none':
            transform_list.append(T.Lambda(
                lambda img: __adjust(img)
            ))
        else:
            raise ValueError('--resize_or_crop {} is not a valid option.'.format(opt.resize_or_crop))

        if opt.isTrain and not opt.no_flip:
            transform_list.append(T.RandomHorizontalFlip())

        transform_list += [T.ToTensor(),
                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform_list = T.Compose(transform_list)

    def __call__(self, img):
        return self.transform_list(img)


def __adjust(img):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if ow == target_width and oh % mult == 0:
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if ow == target_width and oh % mult == 0:
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
