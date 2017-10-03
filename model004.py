# Based on http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr14w-hmlpe.pdf
import os
import time
import tflearn
from tflearn import layers
from tflearn.data_utils import image_preloader as preloader
import numpy as np
from PIL import Image
import util.distances as dis

# For model saving
MODEL_ID = 4
WEIGHTS_FILE = 'weights/model_{:03d}'.format(MODEL_ID)
RESTORE = True

# Configs
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = int(320/2), int(240/2), 3

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
net = layers.fully_connected(net, 512, activation='leaky_relu')
net = layers.dropout(net, 0.5)
net = layers.fully_connected(net, len(Y[0]), activation='leaky_relu')

net = layers.regression(net, loss=dis.corrected_euclidian, optimizer='adam')

# Model
model = tflearn.DNN(net, tensorboard_verbose=2)

if os.path.exists(WEIGHTS_FILE+'.index') and RESTORE:
    print('========== Carregado =========')
    model.load(WEIGHTS_FILE)
    model.fit(X, Y, 100, validation_set=0.1, # 10% as validation
              show_metric=True, batch_size=15, snapshot_step=500,
              snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
    model.save(WEIGHTS_FILE)

else:
    model.fit(X, Y, 100, validation_set=0.1, # 10% as validation
              show_metric=True, batch_size=15, snapshot_step=500,
              snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
    model.save(WEIGHTS_FILE)

from util.result_check import render
for ind in [5,205,405,605,805,905]:
    predict = np.array(model.predict([X[ind]]), dtype=np.uint)
    print(predict)
    print(X.array[ind])
    render(X.array[ind], predict[0], str(ind) + "_predict", (IMAGE_WIDTH, IMAGE_HEIGHT))

from PIL import Image
img = Image.open("./img.png")
img = img.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
img = np.asarray(img)

predict = np.array(model.predict([img]), dtype=np.uint)
render("./img.png", predict[0],"img_predict", (IMAGE_WIDTH, IMAGE_HEIGHT))
