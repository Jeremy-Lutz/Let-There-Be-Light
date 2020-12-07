import numpy as np
import tensorflow as tf
import rawpy
import os


def get_short_image(image_path, amp_ratio):
    """
    This takes the image filepath (as a single element tensor) and returns a numpy array of raw pixel values.
    This is intended only for the short-exposure images, the reference long exposure images use sRGB data
    NOTE: This only works for Bayer array data (Sony camera data), Fujifilm camera uses X-Trans rather than Bayer
    :param image_path: filepath for the image to be processed (as tensor tf.string)
    :param amp_ratio: amplification ratio to multiply image with (as tensor tf.float32)
    :return: an n-d tensor of raw pixel values (tf.float32)
    """
    with rawpy.imread(image_path.numpy().decode()) as raw:
        raw_data = raw.raw_image.copy().astype('float32')
        rows, cols = raw_data.shape

        # this transforms the data into stacked 2x2 patches somehow
        raw_data = raw_data.reshape(rows//2, 2, -1, 2).swapaxes(1, 2).reshape(-1, 2, 2)

        # this transforms the data into [row/2 x col/2 x 4] array with last index being color in the order RGGB
        raw_data = raw_data.reshape(rows//2, -1, 4)

        # subtract off black level, multiply by amp_factor
        raw_data = amp_ratio*np.maximum(raw_data - 512, 0)/(16383 - 512)

    return tf.convert_to_tensor(raw_data)


def get_long_image(image_path):
    """This takes the image filepath (as a single element tensor) and returns a numpy array of raw pixel values.
    This is intended only for long exposure images, which use sRGB data
    :param image_path: filepath (as tensor tf.string)
    :return: an n-d tensor of sRGB pixel values (tf.unit16)
    """
    with rawpy.imread(image_path.numpy().decode()) as raw:
        sRGB_data = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)

    return tf.convert_to_tensor(sRGB_data, dtype=tf.float32)


def process_data(data_path, text_file, batch_sz):
    """

    :return:
    """
    # TODO Collect data from Sony images, return tf.dataset object (probably use batched data loading)
    # TODO Need to process short_exp images by subtracting black level, scale up with proper amp ratio

    in_paths = []
    gt_paths = []
    for line in open(os.path.join(data_path, text_file)).readlines():
        in_path, gt_path = line.split()[0:2]

        in_exp = float(in_path.split('_')[-1][:-5])
        gt_exp = float(gt_path.split('_')[-1][:-5])
        amp_ratio = gt_exp / in_exp

        in_paths.append((os.path.join(data_path, os.path.normpath(in_path)), amp_ratio))
        gt_paths.append(os.path.join(data_path, os.path.normpath(gt_path)))

    in_dataset = tf.data.Dataset.from_generator(lambda: in_paths, (tf.string, tf.float32))
    gt_dataset = tf.data.Dataset.from_tensor_slices(gt_paths)

    in_dataset = in_dataset.map(map_func=lambda x, y: tf.py_function(get_short_image, [x, y], Tout=tf.float32))
    gt_dataset = gt_dataset.map(map_func=lambda x: tf.py_function(get_long_image, [x], Tout=tf.float32))
    in_dataset = in_dataset.batch(batch_sz)
    gt_dataset = gt_dataset.batch(batch_sz)

    in_dataset = in_dataset.prefetch(1)
    gt_dataset = gt_dataset.prefetch(1)

    return in_dataset, gt_dataset

