import numpy as np
import math

from libcore import Img
from libcore import RestructuringMethod


def main():
    img = Img.load_image('../gletscher.jpg')

    # 45 Grad
    rotation_matrix = Img.get_2d_rotation_matrix(np.radians(45))
    translation_vector = np.array([50, 600])

    RestructuringMethod.affine_transform(img,
                                         rotation_matrix,
                                         translation_vector)





if __name__ == "__main__": main()