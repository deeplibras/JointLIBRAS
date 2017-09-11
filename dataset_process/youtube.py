import scipy.io as sio
import numpy as np
import PIL.Image as Image

# data(i).url - string for the youtube weblink for video i
# data(i).videoname - string for the code name of the youtube video
# data(i).locs - 2 by 7 by 100 array containing 2D locations for the ground truth upper body joints.
# Row 1 are x values and Row 2 are y values. Columns are formatted from left to right as:
# Head, Right wrist, Left wrist, Right elbow, Left elbow, Right shoulder and Left shoulder (Person centric).
# data(i).frameids = 1 by 100 array containing the frame indicies which were annotated.
# data(i).label_names - cell array of strings for corresponding body joint labels
# data(i).crop - 1 by 4 array giving the crop bounding box [topx topy botx boty] from the original video
# data(i).scale - value the video should be scaled by
# data(i).imgPath - cell array containing paths to the pre scaled and cropped annotated frames
# data(i).origRes - 1 by 2 array [height,width] resolution of original video
# data(i).isYouTubeSubset - boolean, true if video belongs to the YouTube Subset dataset

FOLDERS = 50-1

URL = 0
FOLDER = 1
POSE = 2
FRAME_IDS = 3
JOINT_LABELS = 4
SHOULD_SCALE = 5
CROP_SIZE = 6
ORI_RESOLUTION = 7
YOUTUBE_SUBSET = 8
X, Y = 0, 1

dataset = sio.loadmat('./data/dataset.mat')
dataset = dataset['data'][0] # contem 50 arrays

# print(dataset[0][FOLDER][0])
# print(dataset[0][POSE][X][0][0])
# print(dataset[0][FRAME_IDS][0][0])

# print('frame001')
# #            video, dado,  x/y, joint, frame
# print(dataset[0]    [POSE] [0]  [0]    [0])
# print(dataset[0][POSE][1][0][0])
# print('\n')

def process(batch_size=50):
    if batch_size > dataset.size:
        raise EOFError('O batch pedido é maior que a quantidade de dados disponível')

    frame_paths = np.empty((batch_size*100), dtype='<U30')
    poses = np.empty((batch_size*100), dtype=list)

    frame_id = 0
    for video in range(0, batch_size):
        folder = str(dataset[video][FOLDER][0])
        for i in range(0, 100):
            # caminho para a imagem
            path = folder + '/frame_{0:06d}.jpg'.format(dataset[video][FRAME_IDS][0][i])
            frame_paths[frame_id] = path

            im = Image.open('data/frames/'+path)

            # lista de poses
            pose = list()
            for joint in range(0, 7):
                joint_x = int(float(dataset[video][POSE][X][joint][i]))
                joint_y = int(float(dataset[video][POSE][Y][joint][i]))
                pose.append([joint_x, joint_y])

            poses[frame_id] = np.array(pose)
            frame_id += 1

    return frame_paths, poses
