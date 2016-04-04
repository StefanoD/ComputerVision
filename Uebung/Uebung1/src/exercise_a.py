import importlib
import numpy as np
from skimage.data import imread
import matplotlib.pyplot as plt

from libcore import Img

from libcore import RestructuringMethod

def indirect_restructuring(image,
                           transform_matrix,
                           translation_vector,
                           restructuring_method = RestructuringMethod.BilinearInterpolation):
    #Enlarge image for transformation
    # More than 2 dimensions
    #if len(image.shape) > 2:
    #    new_dim = (image.shape[0] * 1.5, image.shape[1] * 1.5, len(image.shape))
    #else:

    new_dim = (image.shape[0] * 1.5, image.shape[1] * 1.5, 3)

    new_image = np.zeros(new_dim)
    new_image = new_image

    x_size = new_image.shape[0]
    y_size = new_image.shape[1]

    for x in range(x_size):
        for y in range(y_size):
            new_coordinates = np.array([x, y])
            new_coordinates = new_coordinates - translation_vector

            new_x = new_coordinates[0]
            new_y = new_coordinates[1]

            if new_x < 0 or new_y < 0 or x > image.shape[0] or y > image.shape[1]:
                new_image[x, y] = 0
            else:
                new_image[x, y, 0] = image[new_x, new_y, 0]
                new_image[x, y, 1] = image[new_x, new_y, 1]
                new_image[x, y, 2] = image[new_x, new_y, 2]

            #print(row, col)
            #print(new_row, new_col)



    # back casting to uint8
    uint8_array = new_image.astype(np.uint8)

    # show picture
    plt.imshow(uint8_array)
    plt.show()