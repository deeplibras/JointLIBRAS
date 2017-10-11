from PIL import Image

def render(image, joint, name, size = (256, 256), is_predict=True, crop=(0,0,0,0)):
    img = Image.open(image)
    if crop != (0,0,0,0):
        img = img.crop(crop)
    img = img.resize(size, Image.ANTIALIAS)
    img.save('result_check/' + str(name) + '.png', 'PNG')
    print(joint)

    for i in range(0, len(joint), 2):
        if (joint[i] > size[0]-1 or joint[i+1] > size[1]-1 or joint[i] < 0 or joint[i+1] < 0):
            print("out of range -> ", name, ": ", i, ",", i+1)
        else:
            img.putpixel((int(joint[i]), int(joint[i+1])), (255, 0, 0))

        # img.putpixel((int(joint[i]+1), int(joint[i+1])), (255, 0, 0))
        # img.putpixel((int(joint[i]+1), int(joint[i+1])), (255, 0, 0))
        # img.putpixel((int(joint[i]+2), int(joint[i+1])), (255, 0, 0))
        # img.putpixel((int(joint[i]+2), int(joint[i+1])), (255, 0, 0))
        # img.putpixel((int(joint[i]-1), int(joint[i+1])), (255, 0, 0))
        # img.putpixel((int(joint[i]-1), int(joint[i+1])), (255, 0, 0))
        # img.putpixel((int(joint[i]-2), int(joint[i+1])), (255, 0, 0))
        # img.putpixel((int(joint[i]-2), int(joint[i+1])), (255, 0, 0))
        #
        #
        # img.putpixel((int(joint[i]), int(joint[i+1])+1), (255, 0, 0))
        # img.putpixel((int(joint[i]), int(joint[i+1])+1), (255, 0, 0))
        # img.putpixel((int(joint[i]), int(joint[i+1])+2), (255, 0, 0))
        # img.putpixel((int(joint[i]), int(joint[i+1])+2), (255, 0, 0))
        # img.putpixel((int(joint[i]), int(joint[i+1])-1), (255, 0, 0))
        # img.putpixel((int(joint[i]), int(joint[i+1])-1), (255, 0, 0))
        # img.putpixel((int(joint[i]), int(joint[i+1])-2), (255, 0, 0))
        # img.putpixel((int(joint[i]), int(joint[i+1])-2), (255, 0, 0))

    if(size[0] < 640):
        img = img.resize((size[0]*4,size[1]*4))

    if(is_predict):
        img.save('result_check/' + str(name) + '.png', 'PNG')
    else:
        img.save('load_data_check/' + str(name) + '.png', 'PNG')

    print(str(name) + 'criado')
