from __future__ import division, print_function, absolute_import
import os

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader as preloader
# from process import process
import numpy as np
from PIL import Image

# For model saving
MODEL_ID = 5
WEIGHTS_FILE = 'weights/model_{:03d}'.format(MODEL_ID)

IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256

print("Preloading Images...")
X, _ = preloader('images.txt', image_shape=(IMAGE_WIDTH, IMAGE_HEIGHT), mode='file', categorical_labels=False)

print("Reestructing expected results...")
Y = np.load('joints.npy')
for i, jo in enumerate(Y):
    img = Image.open(X.array[i])

    width, height = img.size
    scales = [ IMAGE_WIDTH / width, IMAGE_HEIGHT / height]

    Y[i][::2] = jo[::2] * scales[0]
    Y[i][1::2] = jo[1::2] * scales[1]

network = input_data(shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 12, activation='relu')
network = regression(network, optimizer='momentum', loss='mean_square', learning_rate=0.001)

model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=0)

if os.path.exists(WEIGHTS_FILE+'.index'):
    model.load(WEIGHTS_FILE)
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='Alex')
    model.save(WEIGHTS_FILE)
else:
    model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='Alex')
    model.save(WEIGHTS_FILE)

print(np.array(model.predict([X[95]]), dtype=np.uint))
print(np.array([Y[95]]))
print("============")
print(np.array(model.predict([X[10]]), dtype=np.uint))
print(np.array([Y[10]]))
