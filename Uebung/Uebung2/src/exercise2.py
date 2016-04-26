import numpy as np
from scipy.misc import imsave

from libcore import Img
from libcore import DistortionCorrection, DistortionCorrectionPoint

image_path = '../schraegbild_tempelhof.jpg'


def main():
    b1()


def b1():
    image = Img.load_image(image_path)

    target_image_size_height = 900
    target_image_size_witdh = 600

    new_x_size = target_image_size_witdh*3
    new_y_size = target_image_size_height*3

    new_image = np.zeros((new_x_size, new_y_size, 3))

    # links oben
    pass_point_1_x = 344.0
    pass_point_1_y = 334.0
    target_point_1_x = 0.0
    target_point_1_y = 0.0

    # links unten
    pass_point_2_x = 300.0
    pass_point_2_y = 456.0
    target_point_2_x = 0.0
    target_point_2_y = target_image_size_height

    pass_point_3_x = 694.0
    pass_point_3_y = 432.0
    #rechts unten
    #pass_point_3_x = 690.0
    #pass_point_3_y = 460.0
    target_point_3_x = target_image_size_witdh
    target_point_3_y = target_image_size_height

    #recht oben
    pass_point_4_x = 548.0
    pass_point_4_y = 330.0
    target_point_4_x = target_image_size_witdh
    target_point_4_y = 0.0

    points = [DistortionCorrectionPoint(344.0, 344.0, 0.0, 0.0),  # links oben
                DistortionCorrectionPoint(300.0, 456.0, 0.0, target_image_size_height),  # links unten
                DistortionCorrectionPoint(694.0, 432.0, target_image_size_witdh, target_image_size_height),
                DistortionCorrectionPoint(548.0, 330.0, target_image_size_witdh, 0.0)] # rechts unten

    new_image = DistortionCorrection.distortion_correction(points, image, new_image)

    imsave("../images/test.jpg", new_image)

if __name__ == "__main__": main()


