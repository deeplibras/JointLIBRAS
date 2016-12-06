from PIL import Image
import numpy as np

def convert_image(filename):
    image = Image.open(filename)
    height, width =  image.size
    pixels = image.load()

    data = []
    for y in range(width):
        for x in range(height):
            r, g, b = pixels[x, y];

            # Converting RGB to grayscale by LUMINOSITY
            # Luminosity: More contrast
            # Average: Same as the original image or less contrast
            # Lightness: Less contrast
            luminosity_grayscale = round(r * 0.21 + g * 0.72 + b * 0.07);

            # grays = [' ','.',':','-','=','+','*','#','%','@']
            # datastring += " "+ str((grays[luminosity_grayscale]/25.5)-1)
            # if(x == width-1):
            #     datastring += "\n"
            print(luminosity_grayscale)
            data.append(luminosity_grayscale)

    return data

# # TODO: Try to make a .h5(HDF5) to make it faster and smaller(maybe)
# # Is being faster to "raster" image to array "on the fly" than write/read from a text file
# def data2file(data, index = 0):
#     data = str(data).replace('[','')
#     data = data.replace(']','')
#     data = data.replace(')]','')
#     data = data.replace('), ','\n')
#     data = data.replace('(','')
#     data = data.replace(')','')
#
#     filename = 'input' + str(index) + '.in'
#     out = file(filename, 'w')
#     out.write(str(data))
#
# def file2data(filename):
#     data = []
#     inp = file(filename, 'r')
#     for line in inp:
#         data.append(line.split(', '))
#     return data

print(str(convert_image('image.jpg')))
