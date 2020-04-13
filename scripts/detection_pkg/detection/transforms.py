"""
Transforms described in https://arxiv.org/abs/1512.02325.
Adapted from https://github.com/dmlc/gluon-cv
"""
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from  gluoncv.data.transforms import image as timage
from  gluoncv.data.transforms import experimental


class SSDDefaultTransform(object):
    """Default SSD validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        img = mx.nd.image.to_tensor(img) # Converts from 0-255 to 0-1
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img


class FasterRCNNDefaultTransform(object):
    """Default Faster-RCNN validation transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """

    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, src):
        """Apply transform to validation image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        # im_scale = h / float(img.shape[0])
        img = mx.nd.image.to_tensor(img)  # Converts from 0-255 to 0-1
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img