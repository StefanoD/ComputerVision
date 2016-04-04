import importlib
import numpy as np
from skimage.data import imread
import matplotlib.pyplot as plt

from libcore import Img

from libcore import RestructuringMethod


def test_indirekte_Umbildung(image, ):



    # More than 2 dimensions
    #if len(image.shape) > 2:
    #    new_dim = (image.shape[0] * 1.5, image.shape[1] * 1.5, len(image.shape))
    #else:
    new_dim = (image.shape[0] * 1.5, image.shape[1] * 1.5)

    new_image = np.zeros(new_dim)
    new_image = new_image.astype(np.float32)




    # back casting to uint8
    uint8_array = image.astype(np.uint8)
    #print uint8_array
    # show picture
    plt.imshow(uint8_array)
    plt.show()




def indirect_restructuring(image,
                           transform_matrix,
                           translation_vector,
                           restructuring_method = RestructuringMethod.BilinearInterpolation):
    #Enlarge image for transformation
    # More than 2 dimensions
    #if len(image.shape) > 2:
    #    new_dim = (image.shape[0] * 1.5, image.shape[1] * 1.5, len(image.shape))
    #else:

    print image.shape

    new_dim = (image.shape[0] * 1.5, image.shape[1] * 1.5, 3)

    new_image = np.zeros(new_dim)
    new_image = new_image

    x_size = new_image.shape[0]
    y_size = new_image.shape[1]

    for row in range(x_size):
        for col in range(y_size):
            new_row = row - translation_vector[0]
            new_col = col - translation_vector[1]

            tmp = np.array([new_row, new_col])


            if new_row < 0 or new_col < 0 or row >= image.shape[0] or col >= image.shape[1]:
                new_image[row, col] = 0
            else:
                new_image[row, col] = image[new_row, new_col, 0]
                new_image[row, col] = image[new_row, new_col, 1]
                new_image[row, col] = image[new_row, new_col, 2]

            #print(row, col)
            #print(new_row, new_col)




    # back casting to uint8
    uint8_array = new_image.astype(np.uint8)
    #print uint8_array
    # show picture
    plt.imshow(uint8_array)
    plt.show()