#!/usr/bin/python3

import numpy as np
from exercise_a import *
from libcore import Img
import math


#theta = np.deg2rad(0)

def main():
    img = Img.load_image('../gletscher.jpg')

    # 45 Grad
    rotation_matrix = Img.get_2d_rotation_matrix(math.pi/4.0)
    translation_matrix = np.array([50, 600])

    indirect_restructuring(img,
                           rotation_matrix,
                           translation_matrix)



if __name__ == "__main__": main()
