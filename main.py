import tensorflow as tf
from u_net import U_Net
from preprocessing import process_data
import numpy as np
import random
import time
import scipy
from matplotlib import pyplot as plt

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.batch_size = 6

        self.u_net = U_Net(12)

    def call(self, im_data):

        im_data = self.u_net(im_data)

        return tf.nn.depth_to_space(im_data, 2)

    def loss(self, predicted_output, ground_truth):

        return tf.reduce_sum(tf.math.abs(predicted_output-ground_truth))

    def accuracy(self, raw_input, ground_truth):

        raw_input = np.minimum(raw_input, 1.)

        ssim = tf.image.ssim(raw_input, ground_truth, 1)
        psnr = tf.image.psnr(raw_input, ground_truth, 1)
        return psnr, ssim


def train(model, in_dataset, gt_dataset):
    checkpoint_dir = "/tmp/training_checkpoints"
    i = 0
    for in_images in in_dataset:
        gt_images = gt_dataset.next()

        # crop out random 512 x 512 patch
        patch_dim = 512

        max_row = tf.shape(in_images)[1].numpy() - patch_dim#1424,2144
        max_col = tf.shape(in_images)[2].numpy() - patch_dim

        row_num = random.randint(0, max_row)
        col_num = random.randint(0, max_col)

        in_images = in_images[:, row_num:row_num+patch_dim, col_num:col_num+patch_dim, :]
        gt_images = gt_images[:, row_num*2:(row_num+patch_dim)*2, col_num*2:(col_num+patch_dim)*2, :]

        # rotate a random number of times
        rot_num = int(random.random()*4)
        in_images = tf.image.rot90(in_images, rot_num)
        gt_images = tf.image.rot90(gt_images, rot_num)

        with tf.GradientTape() as tape:
            pred_images = model.call(in_images)
            loss = model.loss(pred_images, gt_images)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"completed {i}th batch")
        if i % 100 == 0 and i != 0:
            pred_images = model.call(in_images)
            val_psnr, val_ssim = model.accuracy(pred_images, gt_images)
            print(f"After batch {i}, PSNR: {val_psnr}, SSIM: {val_ssim}")
            a=tf.keras.preprocessing.image.array_to_img(255*np.minimum(pred_images[0],1.))
            a.show()
        i+=1


def test(model, in_dataset, gt_dataset):
    ssim_vals = []
    psnr_vals = []

    for in_images in in_dataset:
        gt_images = gt_dataset.next()
        psnr, ssim = model.accuracy(model.call(in_images), gt_images)
        ssim_vals.append(ssim)
        psnr_vals.append(psnr)

    return tf.reduce_mean(psnr_vals), tf.reduce_mean(ssim_vals)


def main():

    # TODO: train, test the model train for no more than 2000 epochs for now
    model = Model()
    data_path = "F:\\Final Project Data\\Sony"

    train_file = "Sony_train_list.txt"
    test_file = "Sony_test_list.txt"
    val_file = "Sony_val_list.txt"  # filename for validation dataset

    train_in_images, train_gt_images = process_data(data_path, train_file, model.batch_size)
    test_in_images, test_gt_images = process_data(data_path, test_file, model.batch_size)
    val_in_images, val_gt_images = process_data(data_path, val_file, model.batch_size)

    epochs = 2000

    for i in range(epochs):
        start_time = time.time()
        train(model, train_in_images.as_numpy_iterator(), train_gt_images.as_numpy_iterator())
        end_time = time.time()
        print(f"Completed epoch #{i} after {end_time-start_time} seconds")
        if i % 100 == 0 and i != 0:
            val_psnr, val_ssim = test(model, val_in_images.as_numpy_iterator(), val_gt_images.as_numpy_iterator())
            print(f"After epoch {i}, PSNR: {val_psnr}, SSIM: {val_ssim}")

    psnr, ssim = test(model, test_in_images.as_numpy_iterator(), test_gt_images.as_numpy_iterator())

    print(f"Mean PSNR: {psnr.numpy()}")
    print(f"Mean SSIM: {ssim.numpy()}")


if __name__ == '__main__':
    #tf.keras.backend.set_floatx('float16')
    main()
