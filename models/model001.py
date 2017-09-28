import os
import time
import tflearn
from tflearn import layers
from tflearn.data_utils import image_preloader as preloader
import numpy as np
from PIL import Image
import util.distances as dis

# For model saving
MODEL_ID = 1
WEIGHTS_FILE = 'weights/model_{:03d}'.format(MODEL_ID)
RESTORE = True

# Configs
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = 320, 240, 3

X, _ = preloader('images.txt', image_shape=(IMAGE_WIDTH, IMAGE_HEIGHT), mode='file', categorical_labels=False)

Y = np.load('poses.npy')
for i, jo in enumerate(Y):
    img = Image.open(X.array[i])

    width, height = img.size
    scales = [ IMAGE_WIDTH / width, IMAGE_HEIGHT / height]

    Y[i][::2] = jo[::2] * scales[0]
    Y[i][1::2] = jo[1::2] * scales[1]

# INPUT LAYER
net = layers.input_data([None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

# CONVOLUTIONAL 001
net = layers.conv_2d(net, 32, 5, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 5)
net = layers.dropout(net, 0.5)

# CONVOLUTIONAL 002
net = layers.conv_2d(net, 32, 3, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 3)
net = layers.dropout(net, 0.5)

# CONVOLUTIONAL 003
net = layers.conv_2d(net, 16, 3, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 3)
net = layers.dropout(net, 0.5)

# FULLY CONNECTED
net = layers.fully_connected(net, 512, activation='relu')
net = layers.dropout(net, 0.5)
net = layers.fully_connected(net, 512, activation='relu', regularizer="L1")
net = layers.dropout(net, 0.5)
net = layers.fully_connected(net, 16, activation='relu')

# REGRESSION
net = layers.regression(net, loss=dis.euclidian_2_2, optimizer='adam')

# DNN MODEL
model = tflearn.DNN(net, tensorboard_verbose=1)

# if os.path.exists(WEIGHTS_FILE+'.index') and RESTORE:
#     print('========== Carregado =========')
#     model.load(WEIGHTS_FILE)
#     model.fit(X, Y, 50, validation_set=0.1, # 10% as validation
#               show_metric=True, batch_size=15, snapshot_step=200,
#               snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
#     model.save(WEIGHTS_FILE)
#
# else:
#     model.fit(X, Y, 50, validation_set=0.1, # 10% as validation
#               show_metric=True, batch_size=15, snapshot_step=200,
#               snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
#     model.save(WEIGHTS_FILE)

from util.result_check import render
for ind in range(1000, 1010):
    predict = np.array(model.predict([X[ind]]), dtype=np.uint)
    print(predict)
    print(X.array[ind])
    # ground  = np.array([Y[ind]])
    render(X.array[ind], predict[0], str(ind) + "_predict", (IMAGE_WIDTH, IMAGE_HEIGHT))
    # render(X.array[ind], ground[0], str(ind) + "_ground")
