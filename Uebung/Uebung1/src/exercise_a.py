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

            if restructuring_method == RestructuringMethod.NearestNeighbor:
                new_x, new_y = nearest_neighboor(new_x, new_y)

            if new_x > 0 and new_y > 0 and new_x < image.shape[0] and new_y < image.shape[1]:
                if restructuring_method == RestructuringMethod.BilinearInterpolation:
                    new_image[x, y, 0] = bilinear_interpolation(image[:, :, 0], new_x, new_y)
                    new_image[x, y, 1] = bilinear_interpolation(image[:, :, 1], new_x, new_y)
                    new_image[x, y, 2] = bilinear_interpolation(image[:, :, 2], new_x, new_y)
                else:
                    new_image[x, y, 0] = image[new_x, new_y, 0]
                    new_image[x, y, 1] = image[new_x, new_y, 1]
                    new_image[x, y, 2] = image[new_x, new_y, 2]


    # back casting to uint8
    uint8_array = new_image.astype(np.uint8)

    # show picture
    plt.imshow(uint8_array)
    plt.show()

def bilinear_interpolation(image, x, y):
    x_left = int(x)
    x_right = int(x + 1)

    y_upper = int(y)
    y_lower = int(y + 1)

    # Because we added 1 on x and y, we could possible be over
    # the range of the image
    image_x_max_index = image.shape[0] - 1
    image_y_max_index = image.shape[1] - 1

    if (x_right > image_x_max_index or y_lower > image_y_max_index):
        return image[x, y]

    # calculate areas
    a1 = (x - x_left) * (y - y_upper)
    a2 = (x_right - x) * (y - y_upper)
    a3 = (x - x_left) * (y_lower - y)
    a4 = (x_right - x) * (y_lower - y)

    grey_value_left_upper = image[x_left, y_upper]
    grey_value_right_upper = image[x_right, y_upper]

    grey_value_left_lower = image[x_left, y_lower]
    grey_value_right_lower = image[x_right, y_lower]

    bilinear_interpolated_gray_value = grey_value_left_upper * a4 + grey_value_right_upper * a3 + \
                                       grey_value_left_lower * a2 + grey_value_right_lower * a1

    return bilinear_interpolated_gray_value


def nearest_neighboor(x, y):
    # round coordinates
    new_x = int(x + 0.5)
    new_y = int(y + 0.5)

    return new_x, new_y