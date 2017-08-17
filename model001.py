# Based on http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr14w-hmlpe.pdf
import os
import tflearn
from tflearn import layers
from tflearn.data_utils import image_preloader

# For model saving
MODEL_ID = 1
WEIGHTS_FILE = 'weights/model_{:03d}'.format(MODEL_ID)

# Configs
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = 640, 480, 3
X, _ = image_preloader(IMAGES_PATH, [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])

# Network
net = layers.input_data([None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])

net = layers.conv_2d(net, 15, 32, padding='valid')
net = layers.max_pool_2d(net, 32)

net = layers.conv_2d(net, 7, 16, padding='valid')
net = layers.max_pool_2d(net, 16)

net = layers.conv_2d(net, 7, 16, padding='valid')
net = layers.max_pool_2d(net, 16)

net = layer.regression(net, loss='mean_square')

# Model
model = tflearn.DNN(net)

if os.path.exists(WEIGHTS_FILE):
    model.load(WEIGHTS_FILE)
else:
    model.fit(X, Y, 100, validation_set=0.1) # 10% as validation
    model.save(WEIGHTS_FILE)
