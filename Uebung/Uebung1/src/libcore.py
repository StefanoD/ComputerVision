from skimage.data import imread
import numpy as np
from enum import Enum
import math

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
    def get_2d_rotation_matrix(degrees):
        rotation_matrix = np.empty((2, 2))

        rotation_matrix[0, 0] = math.cos(degrees)
        rotation_matrix[0, 1] = math.sin(degrees)
        rotation_matrix[1, 0] = -math.sin(degrees)
        rotation_matrix[1, 1] = math.cos(degrees)

        return rotation_matrix

    @staticmethod
    def get_x_3d_rotation_matrix(degrees):
        """Rotation through x axis"""

        rotation_matrix = np.empty((3, 3))

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
