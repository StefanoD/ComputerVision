#!/usr/bin/python3

import numpy as np
from exercise_a import *
from libcore import Img


#theta = np.deg2rad(0)

def main():
    img = Img.load_image('../gletscher.jpg')

    #test_indirekte_Umbildung(img)



    #rotation_matrix = Img.get_2d_rotation_matrix(30)
    indirect_restructuring(image=img, translation_vector=np.array([50, 50]),transform_matrix=None)



if __name__ == "__main__": main()
