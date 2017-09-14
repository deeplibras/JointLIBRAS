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
ONLY_TEST = False

# Configs
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = 720, 405, 3

X, _ = preloader('images.txt', image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='file', categorical_labels=False)

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

net = layers.conv_2d(net, 64, 10, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 10)

net = layers.conv_2d(net, 32, 6, padding='valid', activation='relu')
net = layers.conv_2d(net, 32, 6, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 4)

net = layers.conv_2d(net, 16, 4, padding='valid', activation='relu')
net = layers.conv_2d(net, 16, 4, padding='valid', activation='relu')
net = layers.max_pool_2d(net, 2)

# net = layers.normalization.l2_normalize(net, 0)
net = layers.flatten(net)

net = layers.fully_connected(net, 2048, activation='relu')
net = layers.dropout(net, 0.25)
net = layers.fully_connected(net, 1024, activation='relu')
net = layers.dropout(net, 0.25)
net = layers.fully_connected(net, 18, activation='relu')

net = layers.regression(net, loss='mean_square', optimizer='adam')

# Model
model = tflearn.DNN(net, tensorboard_verbose=1)

if os.path.exists(WEIGHTS_FILE+'.index'):
    print('========== Carregado =========')
    model.load(WEIGHTS_FILE)
    if ONLY_TEST:
        print("=== test ===")
        print(np.array(model.predict([X[10]]), dtype=np.uint))
        print(np.array([Y[10]], dtype=np.uint))
    else:
        model.fit(X, Y, 250, validation_set=0.1, # 10% as validation
                  show_metric=True, snapshot_step=200, batch_size=10,
                  snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
        model.save(WEIGHTS_FILE)

else:
    model.fit(X, Y, 250, validation_set=0.1, # 10% as validation
              show_metric=True, snapshot_step=200, batch_size=10,
              snapshot_epoch=False, run_id=WEIGHTS_FILE+ '::' +str(int(time.time())))
    model.save(WEIGHTS_FILE)

if ONLY_TEST == False:
    from util.result_check import render
    for ind in range(0, 10):
        predict = np.array(model.predict([X[ind]]), dtype=np.uint)
        # ground  = np.array([Y[ind]])
        render(X.array[ind], predict[0], str(ind) + "_predict", (IMAGE_WIDTH, IMAGE_HEIGHT))
        # render(X.array[ind], ground[0], str(ind) + "_ground")
