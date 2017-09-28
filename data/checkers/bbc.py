from PIL import Image
import numpy as np

poses = np.load("../poses.npy")
names = open("../images.txt").readlines()

for i in range(10):
    name = names[i]
    im = Image.open(".." + name[:-3])
    pose = poses[i]

    for xy in range(0, len(pose), 2):
        im.putpixel((int(pose[xy]), int(pose[xy+1])), (255,0,0))
        print(int(pose[xy]), int(pose[xy+1]))

    im.save("../load_data_check/bbc"+str(i)+".png")
    print(str(i) + " criado!")
