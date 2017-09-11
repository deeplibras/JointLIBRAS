# Based on http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr14w-hmlpe.pdf
import os
import time
import tflearn
from tflearn import layers
from tflearn.data_utils import image_preloader as preloader
import numpy as np
from PIL import Image

# For model saving
MODEL_ID = 3
WEIGHTS_FILE = 'weights/model_{:03d}'.format(MODEL_ID)

# Configs
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = 256, 256, 3

X, _ = preloader('images.txt', image_shape=(IMAGE_WIDTH, IMAGE_HEIGHT), mode='file', categorical_labels=False)

# paths, joints = process(1)
# X = list()
# for path in paths:
#     X.append(np.array(Image.open('data/frames/' + path)))
# X = np.array(X, dtype=np.float)
# X = X.reshape((-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
# X = X.reshape((-1, IMAGE_WIDTH, IMAGE_HEIGHT))

Y = np.load('joints.npy')
for i, jo in enumerate(Y):
    img = Image.open(X.array[i])

    width, height = img.size
    scales = [ IMAGE_WIDTH / width, IMAGE_HEIGHT / height]

    Y[i][::2] = jo[::2] * scales[0]
    Y[i][1::2] = jo[1::2] * scales[1]

# for index, image in enumerate(X.array[0:100]):
#     img = Image.open(image)
#     img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
#
#     for i in range(0, 12, 2):
#         img.putpixel((int(Y[index][i]), int(Y[index][i+1])), (255, 0, 0))
#
#     print(str(index) + 'criado')
#     img.save('./teste/' + str(index) + '.png', 'PNG')


# _, joints = process()
# Y = list()
# for joint in joints:
#     Y.append(joint.flatten())
# Y = np.array(Y)
# print(Y[0])


# Network
net = layers.input_data([None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])

net = layers.conv_2d(net, 80, 10, strides=4, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 5, strides=2)

net = layers.conv_2d(net, 32, 5, padding='valid', activation='relu')
net = layers.conv_2d(net, 32, 5, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 3)

net = layers.conv_2d(net, 16, 3, padding='valid', activation='relu')
net = layers.conv_2d(net, 16, 3, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 3)

net = layers.normalization.l2_normalize(net, 0)

net = layers.fully_connected(net, 2048, activation='relu')
net = layers.dropout(net, 0.5)
net = layers.fully_connected(net, 1024, activation='relu', regularizer="L1")
net = layers.dropout(net, 0.5)
net = layers.fully_connected(net, 12, activation='relu')

net = layers.regression(net, loss='mean_square', optimizer='adam')

# Model
model = tflearn.DNN(net, tensorboard_verbose=1)

if os.path.exists(WEIGHTS_FILE+'.index'):
    print('========== Carregado =========')
    model.load(WEIGHTS_FILE)
    model.fit(X, Y, 1, validation_set=0.1, # 10% as validation
              show_metric=True, batch_size=50, snapshot_step=200,
              snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
    model.save(WEIGHTS_FILE)

else:
    model.fit(X, Y, 1, validation_set=0.1, # 10% as validation
              show_metric=True, batch_size=50, snapshot_step=200,
              snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
    model.save(WEIGHTS_FILE)

from teste import render
for ind in range(0, 10):
    predict = np.array(model.predict([X[ind]]), dtype=np.uint)
    # ground  = np.array([Y[ind]])
    render(X.array[ind], predict[0], str(ind) + "_predict")
    # render(X.array[ind], ground[0], str(ind) + "_ground")
