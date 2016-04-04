#!/usr/bin/python3

import numpy as np
from exercise_a import *
from libcore import Img
import math


#theta = np.deg2rad(0)

def main():
    img = Img.load_image('../gletscher.jpg')

    #test_indirekte_Umbildung(img)


    # 45 Grad
    rotation_matrix = Img.get_2d_rotation_matrix(math.pi/4.0)

    indirect_restructuring(image=img,
                           transform_matrix=rotation_matrix,
                           translation_vector=np.array([50, 600]))



if __name__ == "__main__": main()
