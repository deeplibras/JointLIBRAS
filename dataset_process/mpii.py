import scipy.io as sio
import os.path
import numpy as np

dataset = sio.loadmat('../dataset_mpii/dataset.mat')
dataset = dataset['RELEASE']['annolist'][0][0] # abre apenas o que preciso

#   release      lista            info     id
#x['RELEASE']['annolist'][0][0]['image'][0][2][0][0][0][0]

#    release     lista             info        id                        info           joint x/y/jointid/isvisible
#x['RELEASE']['annolist'][0][0]['annorect'][0][10]['annopoints'][0][0]['point'][0][0][0][15][0]

#joint ids
# 0 - r ankle,
# 1 - r knee,
# 2 - r hip,
# 3 - l hip,
# 4 - l knee,
# 5 - l ankle,
# 6 - pelvis,
# 7 - thorax,
# 8 - upper neck,
# 9 - head top,
# 10 - r wrist,
# 11 - r elbow,
# 12 - r shoulder,
# 13 - l shoulder,
# 14 - l elbow,
# 15 - l wrist)


def getImage(index):
    image = None

    try:
        image = '/dataset_mpii/images/' + dataset['image'][0][index][0][0][0][0]
    except IndexError:
        print("erro imagem")
        pass

    return image


def getJoints(index, njoints = range(16)):
    joints = np.zeros((np.array(njoints).size, 2))
    try:
        raw_joints = dataset['annorect'][0][index]['annopoints'][0][0]['point'][0][0][0]

        i = 0
        for j in njoints:
            for k in range(0, 16):
                if(j == raw_joints[k][2]):
                    joints[i][0] = \
                                    raw_joints[j][0]
                    joints[i][1] = \
                                    raw_joints[j][1]
                    i += 1

        joints = joints.flatten()
    except:
        joints = None
        pass

    return joints


# Proccess
images = list()
joints = list()

for i in range(0, 17474):

    image = getImage(i)
    joint = getJoints(i, [10, 11, 12, 13, 14, 15])
    if image is None or joint is None:
        print('Image #' + str(i) + ' ignored.')
        continue
    if not os.path.isfile(image):
        print('Image #' + str(i) + ' ignored. Does not exists')
        continue

    images.append(image + ' 1')
    joints.append(joint)

images = np.array(images)
print(images[0])
joints = np.array(joints)

np.savetxt('../images.txt', images, fmt="%s")
np.save('../joints', joints)
