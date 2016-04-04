import importlib
import numpy as np

from libcore import RestructuringMethod


def indirect_restructuring(image,
                           transform_matrix,
                           translation_vector,
                           restructuring_method = RestructuringMethod.BilinearInterpolation):
    #Enlarge image for transformation
    # More than 2 dimensions
    if len(image.shape) > 2:
        new_dim = (image.shape[0] * 1.5, image.shape[1] * 1.5, len(image.shape))
    else:
        new_dim = (image.shape[0] * 1.5, image.shape[1] * 1.5)

    new_image = np.zeros(new_dim)
    new_image = new_image

    new_image = new_image * transform_matrix