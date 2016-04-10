import numpy as np
from scipy.misc import imsave

from libcore import Img
from libcore import RestructuringMethod


def b1():
    img = Img.load_image('../gletscher.jpg')

    scale_matrix = Img.get_2d_rotation_matrix(np.radians(30))
    translation_vector = np.array([0, 0])

    img_bilinear = RestructuringMethod.affine_transform(img, scale_matrix, translation_vector,
                                                    RestructuringMethod.BilinearInterpolation)
    img_nearest = RestructuringMethod.affine_transform(img, scale_matrix, translation_vector,
                                                   RestructuringMethod.NearestNeighbor)

    imsave("../images/gletscher_b1_bilinear.jpg", img_bilinear)
    imsave("../images/gletscher_b1_nearest.jpg", img_nearest)


def b2():
    img = Img.load_image('../gletscher.jpg')


    scale_matrix = Img.get_2d_scale_matrix(0.7).dot(Img.get_2d_rotation_matrix(np.radians(30)))
    translation_vector = np.array([0, 0])

    img_bilinear = RestructuringMethod.affine_transform(img, scale_matrix, translation_vector,
                                                    RestructuringMethod.BilinearInterpolation)
    img_nearest = RestructuringMethod.affine_transform(img, scale_matrix, translation_vector,
                                                   RestructuringMethod.NearestNeighbor)

    imsave("../images/gletscher_b2_bilinear.jpg", img_bilinear)
    imsave("../images/gletscher_b2_nearest.jpg", img_nearest)


def b3():
    img = Img.load_image('../gletscher.jpg')

    scale_matrix = Img.get_2d_scale_matrix(0.7).dot(Img.get_2d_x_scale_matrix(0.5))
    print(scale_matrix)
    transform_matrix = Img.get_2d_rotation_matrix(np.radians(30)).dot(scale_matrix)

    #scale_matrix = get_2d_x_scale_matrix()
    translation_vector = np.array([0, 0])

    img_bilinear = RestructuringMethod.affine_transform(img, transform_matrix, translation_vector,
                                                    RestructuringMethod.BilinearInterpolation)
    img_nearest = RestructuringMethod.affine_transform(img, transform_matrix, translation_vector,
                                                   RestructuringMethod.NearestNeighbor)

    imsave("../images/gletscher_b3_bilinear.jpg", img_bilinear)
    imsave("../images/gletscher_b3_nearest.jpg", img_nearest)


def main():
    #b1()
    #b2()
    b3()





if __name__ == "__main__": main()