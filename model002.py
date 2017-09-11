import os
import tflearn
from tflearn import layers
from tflearn.data_utils import image_preloader
from process import process
import numpy as np
from PIL import Image

# For model saving
MODEL_ID = 1
WEIGHTS_FILE = 'weights/model_{:03d}'.format(MODEL_ID)

# Configs
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = 419, 236, 3
# X, _ = image_preloader(IMAGES_PATH, [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])

paths, joints = process(1)
X = list()
for path in paths:
    X.append(np.array(Image.open('data/frames/' + path)))
X = np.array(X, dtype=np.float)
X = X.reshape((-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

Y = list()
for joint in joints:
    Y.append(joint.flatten())
Y = np.array(Y)

# model
net = None


model = tflearn.DNN(net, tensorboard_verbose=1)

if os.path.exists(WEIGHTS_FILE+'.index'):
    print('========== Carregado =========')
    model.load(WEIGHTS_FILE)
    model.fit(X, Y, 200, show_metric=True)
    model.save(WEIGHTS_FILE)

else:
    model.fit(X, Y, 200, validation_set=0.1, show_metric=True) # 10% as validation
    model.save(WEIGHTS_FILE)

print(np.array(model.predict([X[95]]), dtype=np.uint))
print(np.array([Y[95]]))
print("============")
print(np.array(model.predict([X[10]]), dtype=np.uint))
print(np.array([Y[10]]))
