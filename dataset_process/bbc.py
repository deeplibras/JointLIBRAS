import scipy.io as sio
import os.path
import numpy as np

dataset = sio.loadmat('../dataset_bbc/dataset.mat')['shortbbcpose'][0]
#       [video][data_id]
#dataset[4]    [7]
# data_id
# 0 - video name
# 1 - type
# 2 - source
# 3 - test_frame
# 4 - test_joints [0][0] is X's array and [0][1] is Y's array
# 5 - train_frames
# 6 - train_joints [0][0] is X's array and [0][1] is Y's array
# each data are wrapped in an array, so, use [0] at final to get the values
X, Y = 0,1
dataset = dataset[0] # now i just want the first video, it has +- 38.000 frames
counter = range(len(dataset[5][0]))

open('../images.txt','w').close()
images = open('../images.txt','a')
poses = list()
for i in counter:
    m_poses = dataset[6]

    joints = list()
    for joint in range(9):
        joints.append(m_poses[X][joint][i])
        joints.append(m_poses[Y][joint][i])
    poses.append(joints)

    images.write('/dataset_bbc/1/' + str(int(dataset[5][0][i])) + ".jpg 0\n")

np.save('../poses', np.array(poses))
images.close()
