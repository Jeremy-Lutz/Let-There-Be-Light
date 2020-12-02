import numpy as np
import tensorflow as tf
import datetime
import rawpy
import glob
import os


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
        raw_data = raw_data.astype(np.float32)
        raw_data = np.maximum(raw_data - 512, 0) / (16383 - 512) #Subtract the black level
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

    # Please change the directory to wherever you're storing your files
    in_dir  = './dataset/Sony/short/'
    gt_dir = './dataset/Sony/long/'

    #Get the image IDs
    in_pathlist = glob.glob(gt_dir + '0*.ARW')
    in_ids = [int(os.path.basename(in_path)[0:5]) for in_path in in_pathlist]
    #Preallocate space for data to speed up
    gt_data = [None]*len(in_ids)
    in_data = [None]*len(in_ids)

    for i in np.random.permutation(len(in_ids)):
        in_path = glob.glob(in_dir + '%05d_00*.ARW' % in_ids[i])
        gt_path = glob.glob(gt_dir + '%05d_00*.ARW' % in_ids[i])
        in_exposure = 0.1
        gt_exposure = float(os.path.basename(str(gt_path))[9:11])
        amp_ratio = min(gt_exposure/in_exposure,300)

        # print(in_path[0])
        in_raw = get_short_image(in_path[0])
        in_data[i] = np.expand_dims(in_raw, axis=0)*amp_ratio
        gt_raw = get_long_image(gt_path[0])
        gt_data[i] = np.expand_dims(np.float32(gt_raw)/65535.0, axis=0)
    return gt_data,in_data
