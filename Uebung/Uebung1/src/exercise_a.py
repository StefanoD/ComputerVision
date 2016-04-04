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

    new_x_size = int(image.shape[0] * 1.5)
    new_y_size = int(image.shape[1] * 1.5)

    new_image = np.zeros((new_x_size, new_y_size, 3))
    new_image = new_image

    print("old dim: ", image.shape, "new_dim", new_image.shape)

    for x in range(new_x_size):
        for y in range(new_y_size):
            new_coordinates = np.array([x, y])

            # First reverse translation
            new_coordinates = new_coordinates - translation_vector

            # Reverse transformation
            new_coordinates = np.dot(new_coordinates, transform_matrix.T)

            new_x = new_coordinates[0]
            new_y = new_coordinates[1]

            if new_x > 0 and new_y > 0 and new_x < image.shape[0] and new_y < image.shape[1]:
                new_image[x, y, 0] = image[new_x, new_y, 0]
                new_image[x, y, 1] = image[new_x, new_y, 1]
                new_image[x, y, 2] = image[new_x, new_y, 2]


    # back casting to uint8
    uint8_array = new_image.astype(np.uint8)

    # show picture
    plt.imshow(uint8_array)
    plt.show()