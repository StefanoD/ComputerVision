import numpy as np
import math

from libcore import Img
from libcore import RestructuringMethod


def main():
    img = Img.load_image('../gletscher.jpg')

    # 45 Grad
    rotation_matrix = Img.get_2d_rotation_matrix(math.pi/4.0)
    translation_vector = np.array([50, 600])

    RestructuringMethod.indirect_restructuring(img,
                           rotation_matrix,
                           translation_vector)





if __name__ == "__main__": main()