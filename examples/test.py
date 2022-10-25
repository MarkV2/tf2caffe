import caffe
from timeit import default_timer as timer
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

def _sigmoid(x):
    return 1. / (1 + np.exp(-x))

def read_image_caffe(input_shape=(256, 256)):

    img1 = cv2.imread('imgs/a.jpg')
    img1 = cv2.resize(img1, input_shape, cv2.INTER_AREA)
    img1 = img1.astype(np.float32) / 127.5 - 1.
    preprocessed_image1 = np.expand_dims(np.transpose(img1, (2, 0, 1)), axis=0)

    return preprocessed_image1

model = 'test.prototxt'
weights = 'test.caffemodel'

caffe_net = caffe.Net(model, weights, caffe.TEST)
caffe_net.blobs['data'].reshape(1, 3, 160, 160)
caffe_net.reshape()

# m = np.array((np.load("image_1.npy") + 1.) * 127.5, np.int32)

# image1 = np.transpose(np.load("image_1.npy")[np.newaxis, :, :, :], (0, 3, 1, 2))
# image2 = np.transpose(np.load("image_2.npy")[np.newaxis, :, :, :], (0, 3, 1, 2))

# image4 = np.concatenate([image1, image2], axis=1)
# print(f'Image shape: {image2.shape}')

img = np.ones((160, 160, 3), dtype=np.float32)
img *= 0.5
image1 = np.transpose(img[np.newaxis, :, :, :], (0, 3, 1, 2))

# start = timer()
caffe_net.blobs['data'].data[...] = image1
caffe_pred = caffe_net.forward()
# print(f'Time taken: {timer() - start}')

for n, i in enumerate(caffe_net.blobs):
    print(n, i)
print()

# layer_name = 'activation_36'
# preds = caffe_net.blobs[layer_name].data
# # param = caffe_net.params[layer_name][0].data
# # np.save('data.npy', caffe_net.params['Block8_6_Branch_1_Conv2d_0a_1x1'][0].data)
# # print(f'Prediction shape: {preds.shape}')

# preds = np.transpose(preds, [0, 2, 3, 1])
# print(f'Output sum: {np.around(np.sum(preds), 8)}')
# print(f'Param sum: {np.around(np.sum(param), 8)}')
print(f'Final prediction: {caffe_pred}')
# print(preds[0][0][0])

# plt.figure(figsize=(8, 8))
# for i in range(25):
#     plt.imshow(preds[0, :, :, i])
#     plt.show()
