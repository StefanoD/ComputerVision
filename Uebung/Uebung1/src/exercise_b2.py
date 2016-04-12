import numpy as np
from scipy.misc import imsave

from libcore import Img
from libcore import RestructuringMethod


image_path = '../gletscher.jpg'
image_path = '../test.jpg'


def b1():
    img = Img.load_image(image_path)

    scale_matrix = Img.get_2d_rotation_matrix(np.radians(30))

    translation_vector = np.array([0, 1])

    img_bilinear = RestructuringMethod.affine_transform(img, scale_matrix, translation_vector,
                                                    RestructuringMethod.BilinearInterpolation)
    img_nearest = RestructuringMethod.affine_transform(img, scale_matrix, translation_vector,
                                                   RestructuringMethod.NearestNeighbor)

    imsave("../images/gletscher_b1_bilinear.jpg", img_bilinear)
    imsave("../images/gletscher_b1_nearest.jpg", img_nearest)



def b2():
    img = Img.load_image(image_path)


    scale_matrix = Img.get_2d_scale_matrix(0.7).dot(Img.get_2d_rotation_matrix(np.radians(30)))
    translation_vector = np.array([0, 0])

    img_bilinear = RestructuringMethod.affine_transform(img, scale_matrix, translation_vector,
                                                    RestructuringMethod.BilinearInterpolation)
    img_nearest = RestructuringMethod.affine_transform(img, scale_matrix, translation_vector,
                                                   RestructuringMethod.NearestNeighbor)

    imsave("../images/gletscher_b2_bilinear.jpg", img_bilinear)
    imsave("../images/gletscher_b2_nearest.jpg", img_nearest)



def b3():
    img = Img.load_image(image_path)

    scale_matrix = Img.get_2d_scale_matrix(0.7).dot(Img.get_2d_x_y_scale_matrix(0.8, 1.2))


    transform_matrix = Img.get_2d_rotation_matrix(np.radians(30)).dot(scale_matrix)

    #scale_matrix = get_2d_x_scale_matrix()
    translation_vector = np.array([0, 0])

    img_bilinear = RestructuringMethod.affine_transform(img, transform_matrix, translation_vector,
                                                    RestructuringMethod.BilinearInterpolation)
    img_nearest = RestructuringMethod.affine_transform(img, transform_matrix, translation_vector,
                                                   RestructuringMethod.NearestNeighbor)

    imsave("../images/gletscher_b3_bilinear.jpg", img_bilinear)
    imsave("../images/gletscher_b3_nearest.jpg", img_nearest)


def b4():
    img = Img.load_image(image_path)

    scale = 0.7

    #scale_matrix = Img.get_2d_x_y_scale_matrix(0.8, 1.2)
    scale_matrix = Img.get_2d_x_y_scale_matrix(0.8, 1.2)
    transform_matrix = Img.get_2d_rotation_matrix(np.radians(30)).dot(scale_matrix)

    transform_matrix = scale*transform_matrix * Img.get_scale_diagonal_matrix(1.5) * Img.get_scale_orthogonal_matrix(0.5)


    translation_vector = np.array([0, 0])

    img_bilinear = RestructuringMethod.affine_transform(img, transform_matrix, translation_vector,
                                                        RestructuringMethod.BilinearInterpolation)
    img_nearest = RestructuringMethod.affine_transform(img, transform_matrix, translation_vector,
                                                       RestructuringMethod.NearestNeighbor)


    imsave("../images/gletscher_b4_bilinear.jpg", img_bilinear)
    imsave("../images/gletscher_b4_nearest.jpg", img_nearest)


def b5():
    img = Img.load_image('../ambassadors.jpg')
    scale = 0.5
    scale_matrix = Img.get_2d_x_y_scale_matrix(9.0, 1.5)

    transform_matrix = Img.get_2d_rotation_matrix(np.radians(30)).dot(scale_matrix)
    transform_matrix = scale*transform_matrix*Img.get_scale_diagonal_matrix(0.4)*Img.get_scale_orthogonal_matrix(0.9)


    translation_vector = np.array([-2000, 0])



    img_bilinear = RestructuringMethod.affine_transform(img, transform_matrix, translation_vector,
                                                        RestructuringMethod.BilinearInterpolation)

    img_nearest = RestructuringMethod.affine_transform(img, transform_matrix, translation_vector,
                                                       RestructuringMethod.NearestNeighbor)

    imsave("../images/ambassadors_b5_bilinear.jpg", img_bilinear)
    imsave("../images/ambassadors_b5_nearest.jpg", img_nearest)


def main():
    b1()
    b2()
    b3()
    b4()
    b5()





if __name__ == "__main__": main()