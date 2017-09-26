# Based on http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr14w-hmlpe.pdf
import os
import time
import tflearn
from tflearn import layers
from tflearn.data_utils import image_preloader as preloader
import numpy as np
from PIL import Image
from util.distances import euclidian

# For model saving
MODEL_ID = 4
WEIGHTS_FILE = 'weights/model_{:03d}'.format(MODEL_ID)

# Configs
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = 720, 405, 3

X, _ = preloader('images.txt', image_shape=(IMAGE_WIDTH, IMAGE_HEIGHT), mode='file', categorical_labels=False)

Y = np.load('poses.npy')
for i, jo in enumerate(Y):
    img = Image.open(X.array[i])

    width, height = img.size
    scales = [ IMAGE_WIDTH / width, IMAGE_HEIGHT / height]

    Y[i][::2] = jo[::2] * scales[0]
    Y[i][1::2] = jo[1::2] * scales[1]

# Network
# rand_weights = tflearn.initializations.uniform(minval=20, maxval=IMAGE_WIDTH)

net = layers.input_data([None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
# net = layers.dropout(net, keep_prob=.8)
#
# net = layers.conv_2d(net, 80, 10, strides=4, padding='valid', activation='relu')
# net = layers.max_pool_2d(net, 5, strides=2)
#
# net = layers.conv_2d(net, 32, 5, padding='valid', activation='relu')
# net = layers.conv_2d(net, 32, 5, padding='valid', activation='relu')
# net = layers.max_pool_2d(net, 3)
#
# # net = layers.normalization.l2_normalize(net, 0)
#
# net = layers.conv_2d(net, 16, 3, padding='valid', activation='relu')
# net = layers.conv_2d(net, 16, 3, padding='valid', activation='relu')
# net = layers.max_pool_2d(net, 3)
#
# # net = layers.flatten(net)
#
# net = layers.fully_connected(net, 2048, activation='relu')
# net = layers.dropout(net, keep_prob=.5)
# net = layers.fully_connected(net, 1024, activation='relu')
# net = layers.dropout(net, keep_prob=.5)
# net = layers.fully_connected(net, 18, activation='relu', weights_init=rand_weights)

net = layers.conv_2d(net, 32, 5, padding='valid', activation='leaky_relu')
net = layers.max_pool_2d(net, 5)
net = layers.dropout(net, 0.5)

net = layers.conv_2d(net, 32, 3, padding='valid', activation='leaky_relu')
net = layers.max_pool_2d(net, 3)
net = layers.dropout(net, 0.5)

net = layers.conv_2d(net, 16, 3, padding='valid', activation='leaky_relu')
net = layers.max_pool_2d(net, 3)
net = layers.dropout(net, 0.5)

net = layers.fully_connected(net, 512, activation='leaky_relu')
net = layers.dropout(net, 0.5)
net = layers.fully_connected(net, 512, activation='linear', regularizer="L1")
net = layers.dropout(net, 0.5)
net = layers.fully_connected(net, 14, activation='linear')

net = layers.regression(net, loss=euclidian, optimizer='adam')

# Model
model = tflearn.DNN(net, tensorboard_verbose=1)

if os.path.exists(WEIGHTS_FILE+'.index'):
    print('========== Carregado =========')
    model.load(WEIGHTS_FILE)
    model.fit(X, Y, 100, validation_set=0.1, # 10% as validation
              show_metric=True, batch_size=15, snapshot_step=200,
              snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
    model.save(WEIGHTS_FILE)

else:
    model.fit(X, Y, 100, validation_set=0.1, # 10% as validation
              show_metric=True, batch_size=15, snapshot_step=200,
              snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
    model.save(WEIGHTS_FILE)

from util.result_check import render
for ind in range(0, 10):
    predict = np.array(model.predict([X[ind]]), dtype=np.uint)
    print(predict)
    print(X.array[ind])
    # ground  = np.array([Y[ind]])
    render(X.array[ind], predict[0], str(ind) + "_predict", (IMAGE_WIDTH, IMAGE_HEIGHT))
    # render(X.array[ind], ground[0], str(ind) + "_ground")
