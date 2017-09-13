from PIL import Image
import numpy as np

poses = np.load("../poses.npy")
names = open("../images.txt").readlines()

for i in range(50,60):
    name = names[i]
    im = Image.open(".." + name[:-3])
    pose = poses[i]

    for xy in range(0, len(pose), 2):
        im.putpixel((pose[xy], pose[xy+1]), (255,0,0))

    im.save("../load_data_check/bbc"+str(i)+".png")

names.close()
