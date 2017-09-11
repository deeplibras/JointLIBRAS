import os
import tflearn
from tflearn import layers
from tflearn.data_utils import image_preloader as preloader
import numpy as np
from PIL import Image

# For model saving
MODEL_ID = 1
WEIGHTS_FILE = 'weights/model_{:03d}'.format(MODEL_ID)

# Configs
IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256

X, _ = preloader('images.txt', image_shape=(IMAGE_WIDTH, IMAGE_HEIGHT), mode='file', categorical_labels=False)

Y = np.load('joints.npy')
for i, jo in enumerate(Y):
    img = Image.open(X.array[i])

    width, height = img.size
    scales = [ IMAGE_WIDTH / width, IMAGE_HEIGHT / height]

    Y[i][::2] = jo[::2] * scales[0]
    Y[i][1::2] = jo[1::2] * scales[1]

# Network
net = layers.input_data([None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

net = layers.conv_2d(net, 32, 5, padding='valid', activation='leaky_relu')
net = layers.max_pool_2d(net, 5)
net = layers.dropout(net, 0.25)

net = layers.conv_2d(net, 32, 3, padding='valid', activation='leaky_relu')
net = layers.max_pool_2d(net, 3)
net = layers.dropout(net, 0.25)

net = layers.conv_2d(net, 16, 3, padding='valid', activation='leaky_relu')
net = layers.max_pool_2d(net, 3)
net = layers.dropout(net, 0.25)

net = layers.fully_connected(net, 512, activation='leaky_relu')
net = layers.dropout(net, 0.25)
net = layers.fully_connected(net, 512, activation='linear', regularizer="L1")
net = layers.dropout(net, 0.25)
net = layers.fully_connected(net, 12, activation='linear')

net = layers.regression(net, loss='mean_square', optimizer='adam')

# Model
model = tflearn.DNN(net, tensorboard_verbose=1)

if os.path.exists(WEIGHTS_FILE+'.index'):
    print('========== Carregado =========')
    model.load(WEIGHTS_FILE)
    # model.fit(X, Y, 1, show_metric=True, validation_set=0.1, batch_size=50)
    # model.save(WEIGHTS_FILE)

else:
    model.fit(X, Y, 1, validation_set=0.1, show_metric=True, batch_size=50) # 10% as validation
    model.save(WEIGHTS_FILE)

from util.result_check import render
for ind in range(0, 10):
    predict = np.array(model.predict([X[ind]]), dtype=np.uint)
    render(X.array[ind], predict[0], str(ind))
