import scipy.io as sio
import os.path
import numpy as np

full_dataset = sio.loadmat('../dataset_bbc/dataset.mat')['shortbbcpose'][0]
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
limit = 5000
X, Y = 0,1

if test:
    testOrTrainFrames = 3
    testOrTrainJoints = 4
else:
    testOrTrainFrames = 5
    testOrTrainJoints = 6

open('../images.txt','w').close()
poses = list()
for di in range(len(full_dataset)):
    dataset = full_dataset[di]

    dataset_size = len(dataset[testOrTrainFrames][0])

    if dataset_size < limit:
        counter = range(dataset_size)
    else:
        counter = range(limit)

    print(counter)
    images = open('../images.txt','a')
    for i in counter:
        m_poses = dataset[testOrTrainJoints]

        joints = list()
        for joint in range(7):
            try:
                joints.append(m_poses[X][joint][i])
                joints.append(m_poses[Y][joint][i])
            except: # ignoring joints not founds (legs)
                joints.append(0)
                joints.append(0)
        poses.append(joints)

        images.write('/dataset_bbc/data/'+str(di+1)+'/' + str(int(dataset[testOrTrainFrames][0][i])) + ".jpg 0\n")

    images.close()
np.save('../poses', np.array(poses))
