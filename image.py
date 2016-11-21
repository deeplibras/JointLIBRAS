from PIL import Image
import numpy as np

def convert_image(filename):
    image = Image.open(filename)
    height, width =  image.size
    pixels = image.load()

    data = []
    for y in range(width):
        for x in range(height):
            data.append(pixels[x, y]) # It Generates an array with R G B values per pixel
    return data

# TODO: Try to make a .h5(HDF5) to make it faster and smaller(maybe)
# Is being faster to "raster" image to array "on the fly" than write/read from a text file
def data2file(data, index = 0):
    data = str(data).replace('[','')
    data = data.replace(']','')
    data = data.replace(')]','')
    data = data.replace('), ','\n')
    data = data.replace('(','')
    data = data.replace(')','')

    filename = 'input' + str(index) + '.in'
    out = file(filename, 'w')
    out.write(str(data))

data2file(convert_image('image.jpg'))
