from PIL import Image
import numpy as np

def convert_image(filename):
    image = Image.open(filename)
    height, width =  image.size
    pixels = image.load()
    luminosity_grayscale = np.zeros((width, height))

    for y in range(width):
        for x in range(height):
            r, g, b = pixels[x, y];

            # Converting RGB to grayscale by LUMINOSITY for more contrast
            luminosity_grayscale[x][y] = round(r * 0.21 + g * 0.72 + b * 0.07);

    return luminosity_grayscale

# Rather than using HDF, it was save in NPY, ONE numpy array in a compressed file
# Still can be improved by use savez_compressed, to save more than one array in a unique file
def data2file(data, index = 0):
    np.save("image"+str(index), data)

def file2data(filename, index = 0):
    return np.load("image"+str(index)+".npy")

print(file2data(data2file(convert_image('image.jpg'))))
