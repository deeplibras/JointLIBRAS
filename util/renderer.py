from PIL import Image

def render(image, joint, name, size = (256, 256), is_predict=True):
    img = Image.open(image)
    img = img.resize(size, Image.ANTIALIAS)

    for i in range(0, 12, 2):
        if (joint[i] > size[0]-1 or joint[i+1] > size[1]-1):
            print("out of range -> ", name, ": ", i, ",", i+1)
        else:
            img.putpixel((int(joint[i]), int(joint[i+1])), (255, 0, 0))

    print(str(name) + 'criado')
    
    if(is_predict):
        img.save('result_check/' + str(name) + '.png', 'PNG')
    else:
        img.save('load_data_check/' + str(name) + '.png', 'PNG')
