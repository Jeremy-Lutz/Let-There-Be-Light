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
        # Input: Two List of images
        # Output: The loss
        return tf.reduce_mean(tf.abs(np.subtract(predicted_output, ground_truth))).numpy()

    def accuracy(self, raw_input, ground_truth):
        # Input: Two List of images
        # Output: The PSNR and SSIM means
        
        def f(x,y):
            return tf.image.psnr(x, y, 1.0)
            
        SSIM = tf.reduce_mean(tf.image.ssim(raw_input, ground_truth, 1.0)).numpy()
        PSNR = tf.reduce_mean([f(x, y) for x, y in zip(raw_input, ground_truth)]).numpy()
        return PSNR, SSIM


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

    return

if __name__ == '__main__':
    main()
