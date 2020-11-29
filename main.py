import tensorflow as tf
from u_net import U_Net
import numpy as np
import scipy


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 100

        self.u_net = U_Net(12)
        self.transpose_conv = tf.keras.layers.Conv2DTranspose(3, 2, strides=2, padding='SAME')

    def call(self, im_data):
        pass

    def loss(self, predicted_output, ground_truth):
        pass

    def accuracy(self, raw_input, ground_truth):
        # TODO: need to return PSNR and SSIM metrics
        pass


def train(model):
    # TODO: we need to iterate through every batch, for each iteration crop to random 512x512 sub-image...
    # rotate, flip randomly as well
    pass

def test(model):
    # TODO: iterate through testing data, acquire accuracy values return average
    pass

def main():
    # TODO: load the tf.dataset object
    # TODO: train, test the model

    pass
