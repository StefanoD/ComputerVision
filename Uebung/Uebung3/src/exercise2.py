import numpy as np
from scipy.misc import imsave

from libcore import Img
from libcore import RestructuringMethod, DistortionCorrection, DistortionCorrectionPoint

image_path = '../schraegbild_tempelhof.jpg'


def main():
    zwei_monitor()


def zwei_monitor():
    mon1 = Img.load_image('../images/Bild1.jpg')
    mon2 = Img.load_image('../images/Bild2.jpg')

    target_image_size_height_1 = 525.0
    target_image_size_witdh_1 = 420.0

    new_image_1 = np.zeros((target_image_size_height_1, target_image_size_witdh_1, 3))

    target_image_size_height_2 = 525.0
    target_image_size_witdh_2 = 420.0

    new_image_2 = np.zeros((target_image_size_height_2, target_image_size_witdh_2, 3))

    points_1 = [DistortionCorrectionPoint(476.0, 578.0, 0.0, 0.0),  # links oben
              DistortionCorrectionPoint(501.0, 964.0, 0.0, target_image_size_height_1),  # links unten
              DistortionCorrectionPoint(742.0, 897.0, target_image_size_witdh_1, target_image_size_height_1),  # rechts unten
              DistortionCorrectionPoint(714.0, 555.0, target_image_size_witdh_1, 0.0)]


    points_2 = [DistortionCorrectionPoint(127.0, 482.0, 0.0, 0.0),  # links oben
          DistortionCorrectionPoint(124.0, 844.0, 0.0, target_image_size_height_2),  # links unten
          DistortionCorrectionPoint(399.0, 838.0, target_image_size_witdh_2, target_image_size_height_2),
          # rechts unten
          DistortionCorrectionPoint(414.0, 486.0, target_image_size_witdh_2, 0.0)]

    new_image_1 = DistortionCorrection.distortion_correction(points_1, mon1, new_image_1)
    new_image_2 = DistortionCorrection.distortion_correction(points_2, mon2, new_image_2)

    imsave("../images/Bild_1_correct.jpg", new_image_1)
    imsave("../images/Bild_2_correct.jpg", new_image_2)

def test_monitor():
    #image = Img.load_image('../images/Bild_A.jpg')
    image = Img.load_image('../images/Bild_A_org.jpg')

    target_image_size_height = 525.0
    target_image_size_witdh = 840.0

    new_image = np.zeros((target_image_size_height, target_image_size_witdh, 3))

    #points = [DistortionCorrectionPoint(96.0, 118.0, 0.0, 0.0),#links oben
    #          DistortionCorrectionPoint(100.0, 191.0, 0.0, target_image_size_height),# links unten
    #          DistortionCorrectionPoint(184.0, 168.0, target_image_size_witdh, target_image_size_height),#rechts unten
    #          DistortionCorrectionPoint(184.0, 108.0, target_image_size_witdh, 0.0)]

    points = [DistortionCorrectionPoint(459.0, 563.0, 0.0, 0.0),  # links oben
              DistortionCorrectionPoint(485.0, 991.0, 0.0, target_image_size_height),  # links unten
              DistortionCorrectionPoint(938.0, 857.0, target_image_size_witdh, target_image_size_height),       # rechts unten
              DistortionCorrectionPoint(935.0, 523.0, target_image_size_witdh, 0.0)]
              #DistortionCorrectionPoint(1077.0, 643.0, target_image_size_witdh +138 , target_image_size_height / 2.0)]

    new_image = DistortionCorrection.distortion_correction(points, image, new_image)

    imsave("../images/Bild_A_correct.jpg", new_image)

if __name__ == "__main__": main()


