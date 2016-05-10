import numpy as np
from scipy.misc import imsave

from libcore import Img
from libcore import DistortionCorrection, DistortionCorrectionPoint, ImageAndPasspoints, StitchMode


def main():
    # zwei_monitor()
    #schachbrett_stiching_kein_ueberlapp()
    zwei_bilder_stiching()



def zwei_bilder_stiching():
    links = Img.load_image('../gGebaeude/links.jpg')
    mitte_links = Img.load_image('../gGebaeude/mitteLinks.jpg')
    mitte_rechts = Img.load_image('../gGebaeude/mitteRechts.jpg')
    rechts = Img.load_image('../gGebaeude/rechts.jpg')


    points_links = [DistortionCorrectionPoint(1215.0, 919.0, 0.0, 0.0),  # links oben
                    DistortionCorrectionPoint(1001.0, 3066.0, 0.0, 650),  # links unten
                    DistortionCorrectionPoint(4439.0, 461.0, 1125, 0) , #rechts oben
                    DistortionCorrectionPoint(4392.0, 3321.0, 1125, 650)]   #rechts unten

    #old rechts oben (3665.0, 466.0, 888, 0.0)
    #old rechts unten(3692.0, 3249.0, 1011, 650)
    points_mitte_links = [DistortionCorrectionPoint(549.0, 496.0, 0.0, 0.0),  # links oben
                    DistortionCorrectionPoint(366.0, 3230.0, 0.0, 650),  # links unten
                    DistortionCorrectionPoint(4175.0, 461.0, 1011, 0.0), #rechts oben #4175 (1011), 461
                    DistortionCorrectionPoint(4261.0, 3249.0, 1011, 650)]  #rechts unten #4261, 3249

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


    stitched_image, new_weight = Img.sticht_images_vignete(stichting_images,StitchMode.MODE_MULTIBAND_BLENDING)



    imsave('../gGebaeude/stitched_image.jpg',stitched_image)


    #imsave('../gGebaeude/new_weight.jpg', new_weight)



def schachbrett_stiching_kein_ueberlapp():
    oben = Img.load_image('../images_schachbrett/stitching/oben.jpg')
    links = Img.load_image('../images_schachbrett/stitching/links.jpg')
    rechts = Img.load_image('../images_schachbrett/stitching/rechts.jpg')

    alles = Img.load_image('../images_schachbrett/stitching/alles.jpg')

    t = 0

    points_oben = [DistortionCorrectionPoint(229.0, 565.0, 0.0 + t, 0.0 + t),  # links oben
                   DistortionCorrectionPoint(214.0, 725.0, 0.0 + t, 540 + t),  # links unten
                   DistortionCorrectionPoint(670.0, 691.0, 1920 + t, 540 + t),  # rechts unten
                   DistortionCorrectionPoint(669.0, 564.0, 1920 + t, 0.0 + t)]

    points_links = [DistortionCorrectionPoint(593.0, 395.0, 0.0 + t, 0.0 + t),  # links oben
                    DistortionCorrectionPoint(731.0, 388.0, 480 + t, 0 + t),  # links unten
                    DistortionCorrectionPoint(587.0, 686.0, 0 + t, 1080 + t),  # rechts unten
                    DistortionCorrectionPoint(728.0, 698.0, 480 + t, 1080 + t)]

    points_rechts = [DistortionCorrectionPoint(16.0, 390.0, 1440 + t, 0.0 + t),
                     DistortionCorrectionPoint(139.0, 400.0, 1920 + t, 0 + t),
                     DistortionCorrectionPoint(24.0, 680.0, 1440 + t, 1080 + t),
                     DistortionCorrectionPoint(147.0, 663.0, 1920 + t, 1080 + t)]

    points_alles = [DistortionCorrectionPoint(379.0, 255.0, 0.0 + t, 0.0 + t),  # links oben
                    DistortionCorrectionPoint(367.0, 524.0, 0.0 + t, 1080 + t),  # links unten
                    DistortionCorrectionPoint(702.0, 515.0, 1920 + t, 1080 + t),  # rechts unten

                    DistortionCorrectionPoint(704.0, 310.0, 1920 + t, 0.0 + t)]

    new_image_oben = DistortionCorrection.distortion_correction(points_oben, oben)
    new_image_links = DistortionCorrection.distortion_correction(points_links, links)
    new_images_rechts = DistortionCorrection.distortion_correction(points_rechts, rechts)
    new_images_alles = DistortionCorrection.distortion_correction(points_alles, alles)

    bilder = [new_images_alles, new_image_oben, new_image_links, new_images_rechts]
    stich = Img.sticht_images_copy(bilder)

    imsave("../images_schachbrett/stitching/stich_correct.jpg", stich)


    # imsave("../images_schachbrett/stitching/oben_correct.jpg", new_image_oben)
    # imsave("../images_schachbrett/stitching/links_correct.jpg", new_image_links)
    # imsave("../images_schachbrett/stitching/rechts_correct.jpg", new_images_rechts)
    # imsave("../images_schachbrett/stitching/alles_correct.jpg", new_images_alles)


def schachbrett():
    image = Img.load_image('../images_schachbrett/Schachbrett_Klein.jpg')
    # translation
    t = 100

    points = [DistortionCorrectionPoint(379.0, 255.0, 0.0 + t, 0.0 + t),  # links oben
              DistortionCorrectionPoint(367.0, 524.0, 0.0 + t, 540 + t),  # links unten
              DistortionCorrectionPoint(702.0, 515.0, 960 + t, 540 + t),  # rechts unten

              DistortionCorrectionPoint(704.0, 310.0, 960 + t, 0.0 + t),
              DistortionCorrectionPoint(373.0, 387.0, 0 + t, 270 + t),
              DistortionCorrectionPoint(560.0, 401.0, 480 + t, 270 + t)]

    tmp_x_max = -1
    tmp_y_max = -1

    for point in points:
        if tmp_x_max < point.target_point_x:
            tmp_x_max = point.target_point_x
        if tmp_y_max < point.target_point_y:
            tmp_y_max = point.target_point_y

    print "Max-X = {0}, Max-y={1}".format(tmp_x_max, tmp_y_max)

    o = 50
    new_image = np.zeros((tmp_y_max + t + o, tmp_x_max + t + o, 3))

    new_image = DistortionCorrection.distortion_correction(points, image, new_image)

    imsave("../images_schachbrett/Schachbrett_Klein_correct.jpg", new_image)


def zwei_monitor():
    mon1 = Img.load_image('../images/Bild1.jpg')
    mon2 = Img.load_image('../images/Bild2.jpg')

    # 262,5
    # 210



    target_image_size_height_1 = 525.0
    target_image_size_width_1 = 420.0

    new_image_1 = np.zeros((target_image_size_height_1 * 2, target_image_size_width_1 * 2, 3))

    target_image_size_height_2 = 525.0
    target_image_size_width_2 = 420.0

    new_image_2 = np.zeros((target_image_size_height_2, target_image_size_width_2, 3))

    # bild kords, 486,771, welt:0, 210

    k = 0

    points_1 = [DistortionCorrectionPoint(476.0, 578.0, 0.0 + k, 0.0 + k),  # links oben
                DistortionCorrectionPoint(501.0, 964.0, 0.0 + k, target_image_size_height_1 + k),  # links unten
                DistortionCorrectionPoint(742.0, 897.0, target_image_size_width_1 + k, target_image_size_height_1 + k),
                # rechts unten
                DistortionCorrectionPoint(714.0, 555.0, target_image_size_width_1 + k, 0.0 + k),
                DistortionCorrectionPoint(486.0, 750.0, 0 + k, 210.0 + k)]

    points_2 = [DistortionCorrectionPoint(127.0, 482.0, 0.0, 0.0),  # links oben
                DistortionCorrectionPoint(124.0, 844.0, 0.0, target_image_size_height_2),  # links unten
                DistortionCorrectionPoint(399.0, 838.0, target_image_size_width_2, target_image_size_height_2),
                # rechts unten
                DistortionCorrectionPoint(414.0, 486.0, target_image_size_width_2, 0.0)]

    new_image_1 = DistortionCorrection.distortion_correction(points_1, mon1, new_image_1)
    # new_image_2 = DistortionCorrection.distortion_correction(points_2, mon2, new_image_2)

    stichted_img = Img.sticht_images(new_image_1, new_image_2)

    imsave("../images/Bild_1_correct.jpg", new_image_1)
    # imsave("../images/Bild_2_correct.jpg", new_image_2)
    # imsave("../images/Bild_stichted.jpg", stichted_img)


def test_monitor():
    # image = Img.load_image('../images/Bild_A.jpg')
    image = Img.load_image('../images/Bild_A_org.jpg')

    target_image_size_height = 525.0
    target_image_size_witdh = 840.0

    new_image = np.zeros((target_image_size_height, target_image_size_witdh, 3))

    # points = [DistortionCorrectionPoint(96.0, 118.0, 0.0, 0.0),#links oben
    #          DistortionCorrectionPoint(100.0, 191.0, 0.0, target_image_size_height),# links unten
    #          DistortionCorrectionPoint(184.0, 168.0, target_image_size_witdh, target_image_size_height),#rechts unten
    #          DistortionCorrectionPoint(184.0, 108.0, target_image_size_witdh, 0.0)]

    points = [DistortionCorrectionPoint(459.0, 563.0, 0.0, 0.0),  # links oben
              DistortionCorrectionPoint(485.0, 991.0, 0.0, target_image_size_height),  # links unten
              DistortionCorrectionPoint(938.0, 857.0, target_image_size_witdh, target_image_size_height),
              # rechts unten
              DistortionCorrectionPoint(935.0, 523.0, target_image_size_witdh, 0.0)]
    # DistortionCorrectionPoint(1077.0, 643.0, target_image_size_witdh +138 , target_image_size_height / 2.0)]

    new_image = DistortionCorrection.distortion_correction(points, image, new_image)

    imsave("../images/Bild_A_correct.jpg", new_image)


if __name__ == "__main__": main()
