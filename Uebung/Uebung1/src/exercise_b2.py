import numpy as np
import math

from libcore import Img
from libcore import RestructuringMethod


def main():
    img = Img.load_image('../gletscher.jpg')

    scale_matrix = Img.get_2d_scale_matrix(0.7)
    translation_vector = np.array([0, 0])

    RestructuringMethod.affine_transform(img,
                                         scale_matrix,
                                         translation_vector)





if __name__ == "__main__": main()