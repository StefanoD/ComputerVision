#!/usr/bin/python3

import numpy as np
import exercise_a
from libcore import Img


def main():
    img = Img.load_image('../gletscher.jpg')

    rotation_matrix = Img.get_2d_rotation_matrix(30)
    exercise_a.indirect_restructuring(img, rotation_matrix, [], [])



if __name__ == "__main__": main()
