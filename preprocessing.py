import numpy as np
import tensorflow as tf
import datetime
import rawpy


def get_short_image(image_path):
    """
    This takes the image filepath and returns a numpy array of raw pixel values.
    This is intended only for the short-exposure images, the reference long exposure images use sRGB data
    NOTE: This only works for Bayer array data (Sony camera data), Fujifilm camera uses X-Trans rather than Bayer
    :param image_path: filepath for the image to be processed
    :return: an n-d array of raw pixel values
    """
    with rawpy.imread(image_path) as raw:
        raw_data = raw.raw_image.copy()
        rows, cols = raw_data.shape

        # this transforms the data into stacked 2x2 patches somehow
        raw_data = raw_data.reshape(rows//2, 2, -1, 2).swapaxes(1, 2).reshape(-1, 2, 2)

        # this transforms the data into [row/2 x col/2 x 4] array with last index being color in the order RGGB
        raw_data = raw_data.reshape(rows//2, -1, 4)

    return raw_data


def get_long_image(image_path):
    """


    :param image_path:
    :return:
    """
    with rawpy.imread(image_path) as raw:
        sRGB_data = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)

    return sRGB_data

def process_data():
    """

    :return:
    """
    # TODO Collect data from Sony images, return tf.dataset object (probably use batched data loading)
    # TODO Need to process short_exp images by subtracting black level, scale up with proper amp ratio
    im_data = None
    amp_ratio = None
    im_data = np.maximum(im_data - 512, 0) / (16383 - 512)

