import numpy as np
from scipy.misc import imsave

from libcore import Img
from libcore import RestructuringMethod

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

    equalisation_matrix1 = np.zeros((8,8))

    equalisation_matrix = np.array([[pass_point_1_x, pass_point_1_y, 1, 0, 0, 0, -target_point_1_x*pass_point_1_x,-target_point_1_x*pass_point_1_y],
                                    [0,0,0,pass_point_1_x,pass_point_1_y,1,-target_point_1_y*pass_point_1_x,-target_point_1_y*pass_point_1_y],
                                    [pass_point_2_x,pass_point_2_y,1,0,0,0,-target_point_2_x*pass_point_2_x,-target_point_2_x*pass_point_2_y],
                                    [0,0,0,pass_point_2_x,pass_point_2_y,1,-target_point_2_y*pass_point_2_x,-target_point_2_y*pass_point_2_y],
                                    [pass_point_3_x,pass_point_3_y,1,0,0,0,-target_point_3_x*pass_point_3_x,-target_point_3_x*pass_point_3_y],
                                    [0,0,0,pass_point_3_x,pass_point_3_y,1,-target_point_3_y*pass_point_3_x,-target_point_3_y*pass_point_3_y],
                                    [pass_point_4_x,pass_point_4_y,1,0,0,0,-target_point_4_x*pass_point_4_x,-target_point_4_x*pass_point_4_y],
                                    [0,0,0,pass_point_4_x,pass_point_4_y,1,-target_point_4_y*pass_point_4_x,-target_point_4_y*pass_point_4_y]])

    target_points = np.transpose([target_point_1_x,target_point_1_y,target_point_2_x,target_point_2_y,target_point_3_x,target_point_3_y,target_point_4_x,target_point_4_y])


    a = (np.linalg.inv(equalisation_matrix)).dot(target_points)

    print (a.shape)

    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    b1 = a[3]
    b2 = a[4]
    b3 = a[5]
    c1 = a[6]
    c2 = a[7]


    for x_old in np.arange(0, image.shape[1]):
        for y_old in np.arange(0, image.shape[0]):
            denominator = (b1*c2 - b2*c1)*y_old + (a2*c1 - a1*c2)*x_old + a1*b2 - a2*b1
            new_x = ((b2 - c2*b3)*y_old + (a3*c2 - a2)*x_old + a2*b3 - a3*b2)/denominator
            new_y = ((b3*c1 - b1)*y_old + (a1 - a3*c1)*x_old + a3*b1 - a1*b3)/denominator

            if new_x > 0 and new_y > 0 and new_x < image.shape[1] and new_y < image.shape[0]:
                new_image[x_old, y_old, 0] = RestructuringMethod.bilinear_interpolation(image[:, :, 0], new_y, new_x)
                new_image[x_old, y_old, 1] = RestructuringMethod.bilinear_interpolation(image[:, :, 1], new_y, new_x)
                new_image[x_old, y_old, 2] = RestructuringMethod.bilinear_interpolation(image[:, :, 2], new_y, new_x)

    imsave("../images/test.jpg", new_image)

if __name__ == "__main__": main()


