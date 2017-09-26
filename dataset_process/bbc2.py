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
test = True
X, Y = 0,1

if test:
    testOrTrainFrames = 3
    testOrTrainJoints = 4
else:
    testOrTrainFrames = 5
    testOrTrainJoints = 6

dataset = dataset[0] # now i just want the first video, it has +- 38.000 frames
counter = range(len(dataset[testOrTrainFrames][0]))

open('../images.txt','w').close()
images = open('../images.txt','a')
poses = list()
for i in counter:
    m_poses = dataset[testOrTrainJoints]

    joints = list()
    for joint in range(9):
        try:
            joints.append(m_poses[X][joint][i], m_poses[Y][joint][i])
        except: # ignoring joints not founds (legs)
            joints.append([0,0])
    poses.append(joints)

    images.write('/dataset_bbc/data/1/' + str(int(dataset[testOrTrainFrames][0][i])) + ".jpg 0\n")

np.save('../poses', np.array(poses))
images.close()
