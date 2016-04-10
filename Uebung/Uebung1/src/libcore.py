from skimage.data import imread
import numpy as np
from enum import Enum
import math
import matplotlib.pyplot as plt

class Img:

    @staticmethod
    def load_image(path, as_grey = False, to_float = True):
        # Load image
        image = imread(path, as_grey)

        if to_float:
            # Convert to floating point matrix
            image = image.astype(np.float32)

        return image

    @staticmethod
    def get_2d_rotation_matrix(rad):
        rotation_matrix = np.zeros((2, 2))

        rotation_matrix[0, 0] = math.cos(rad)
        rotation_matrix[0, 1] = -math.sin(rad)

        rotation_matrix[1, 0] = math.sin(rad)
        rotation_matrix[1, 1] = math.cos(rad)

        return rotation_matrix

    @staticmethod
    def get_2d_scale_matrix(scale):
        scale_matrix = np.zeros((2, 2))

        scale_matrix[0, 0] = scale
        scale_matrix[1, 1] = scale

        return scale_matrix

    @staticmethod
    def get_2d_x_scale_matrix(scale):
        x_scale_matrix = np.zeros((2, 2))

        x_scale_matrix[0, 0] = scale
        x_scale_matrix[1, 1] = 1

        return x_scale_matrix

    @staticmethod
    def get_2d_x_y_scale_matrix(x_scale, y_scale):
        x_scale_matrix = np.zeros((2, 2))

        x_scale_matrix[0, 0] = x_scale
        x_scale_matrix[1, 1] = y_scale

        return x_scale_matrix


    @staticmethod
    def get_x_3d_rotation_matrix(degrees):
        """Rotation through x axis"""

        rotation_matrix = np.zeros((3, 3))

        rotation_matrix[0, 0, 0] = 1

        rotation_matrix[1, 1, 1] = math.cos(degrees)
        rotation_matrix[1, 1, 2] = -math.sin(degrees)

        rotation_matrix[2, 2, 1] = math.sin(degrees)
        rotation_matrix[2, 2, 2] = math.cos(degrees)

        return rotation_matrix



class Transform:

    @staticmethod
    def translate(matrix, trans_vector):
        return matrix + trans_vector


class RestructuringMethod(Enum):
    NearestNeighbor = 1
    BilinearInterpolation = 2


    @staticmethod
    def affine_transform(image,
                         transform_matrix,
                         translation_vector,
                         restructuring_method=BilinearInterpolation):
        from numpy.linalg import inv

        new_x_size = int(image.shape[0] * 1.5)
        new_y_size = int(image.shape[1] * 1.5)

        new_image = np.zeros((new_x_size, new_y_size, 3))

        # Get the inverse matrix for indirect restructuring
        trans_inv = inv(transform_matrix)

        for x in range(new_x_size):
            for y in range(new_y_size):
                new_coordinates = np.array([x, y])

                # First reverse translation
                new_coordinates = new_coordinates - translation_vector

                # Reverse transformation
                new_coordinates = np.dot(new_coordinates, trans_inv)

                new_x = new_coordinates[0]
                new_y = new_coordinates[1]

                if restructuring_method == RestructuringMethod.NearestNeighbor:
                    new_x, new_y = RestructuringMethod.nearest_neighboor(new_x, new_y)

                if new_x > 0 and new_y > 0 and new_x < image.shape[0] and new_y < image.shape[1]:
                    if restructuring_method == RestructuringMethod.BilinearInterpolation:
                        new_image[x, y, 0] = RestructuringMethod.bilinear_interpolation(image[:, :, 0], new_x, new_y)
                        new_image[x, y, 1] = RestructuringMethod.bilinear_interpolation(image[:, :, 1], new_x, new_y)
                        new_image[x, y, 2] = RestructuringMethod.bilinear_interpolation(image[:, :, 2], new_x, new_y)
                    else:
                        new_image[x, y, 0] = image[new_x, new_y, 0]
                        new_image[x, y, 1] = image[new_x, new_y, 1]
                        new_image[x, y, 2] = image[new_x, new_y, 2]

        # back casting to uint8
        return new_image.astype(np.uint8)


    @staticmethod
    def bilinear_interpolation(image, x, y):
        x_left = int(x)
        x_right = int(x + 1)

        y_upper = int(y)
        y_lower = int(y + 1)

        # Because we added 1 on x and y, we could possibly be over
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

    @staticmethod
    def nearest_neighboor(x, y):
        # round coordinates
        new_x = int(x + 0.5)
        new_y = int(y + 0.5)

        return new_x, new_y
