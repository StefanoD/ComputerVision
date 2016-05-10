import numpy as np
from scipy.misc import imsave

from libcore import Img
from libcore import DistortionCorrection, DistortionCorrectionPoint, ImageAndPasspoints, StitchMode


def main():
    zwei_bilder_stiching()



def zwei_bilder_stiching():
    links = Img.load_image('../gGebaeude/links.jpg')
    mitte_links = Img.load_image('../gGebaeude/mitteLinks.jpg')
    mitte_rechts = Img.load_image('../gGebaeude/mitteRechts.jpg')
    rechts = Img.load_image('../gGebaeude/rechts.jpg')

    points_links = [DistortionCorrectionPoint(1215.0, 919.0, 0.0, 0.0),  # links oben
                    DistortionCorrectionPoint(1001.0, 3066.0, 0.0, 650),  # links unten
                    DistortionCorrectionPoint(4439.0, 461.0, 1125, 0),  # rechts oben
                    DistortionCorrectionPoint(4392.0, 3321.0, 1125, 650)]  # rechts unten

    points_mitte_links = [DistortionCorrectionPoint(549.0, 496.0, 0.0, 0.0),  # links oben
                          DistortionCorrectionPoint(366.0, 3230.0, 0.0, 650),  # links unten
                          DistortionCorrectionPoint(4175.0, 461.0, 1011, 0.0),  # rechts oben
                          DistortionCorrectionPoint(4261.0, 3249.0, 1011, 650)]  # rechts unten

    DistortionCorrectionPoint.set_move_to_right_in_array(points_mitte_links, 992)

    points_mitte_rechts = [DistortionCorrectionPoint(505.0, 395.0, 0.0, 0.0),  # links oben
                           DistortionCorrectionPoint(489.0, 3181.0, 0.0, 650),  # links unten
                           DistortionCorrectionPoint(4058.0, 493.0, 990, 0),  # rechts oben
                           DistortionCorrectionPoint(4250.0, 3027.0, 990, 650)]  # rechts unten

    DistortionCorrectionPoint.set_move_to_right_in_array(points_mitte_rechts, 992+1011)

    points_rechts = [DistortionCorrectionPoint(763.0, 771.0, 0.0, 0.0),  # links oben
                     DistortionCorrectionPoint(817.0, 3164.0, 0.0, 650),  # links unten
                     DistortionCorrectionPoint(3272.0, 1191.0, 1169, 0),  # rechts oben
                     DistortionCorrectionPoint(3417.0, 3019.0, 1169, 650)]  # rechts unten

    DistortionCorrectionPoint.set_move_to_right_in_array(points_rechts, 992 + 1011+990)

    stichting_images = [ ImageAndPasspoints(links,points_links),
                         ImageAndPasspoints(mitte_links, points_mitte_links),
                         ImageAndPasspoints(mitte_rechts, points_mitte_rechts),
                         ImageAndPasspoints(rechts, points_rechts)]

    mode = StitchMode.MODE_MULTIBAND_BLENDING
    stitched_image = Img.sticht_images_vignete(stichting_images, mode)
    imsave('../gGebaeude/stitched_image_{}.jpg'.format(mode), stitched_image)

    mode = StitchMode.MODE_SUM
    stitched_image = Img.sticht_images_vignete(stichting_images, mode)
    imsave('../gGebaeude/stitched_image_{}.jpg'.format(mode), stitched_image)

    mode = StitchMode.MODE_STRONGEST
    stitched_image = Img.sticht_images_vignete(stichting_images, mode)
    imsave('../gGebaeude/stitched_image_{}.jpg'.format(mode), stitched_image)


if __name__ == "__main__": main()
