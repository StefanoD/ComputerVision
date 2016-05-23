# -*- coding: utf-8 -*-

from skimage.data import imread
from skimage import transform as tf
from scipy.ndimage.filters import gaussian_filter

from scipy import ndimage

import numpy as np
import math

from scipy.misc import imsave

DEBUG = True

class StitchMode:
    MODE_STRONGEST = "MODE_STRONGEST"
    MODE_SUM = "MODE_SUM"
    MODE_MULTIBAND_BLENDING = "MODE_MULTIBAND_BLENDING"

class Img:
    @staticmethod
    def load_image(path, as_grey=False, to_float=True):

        if DEBUG:
            im = imread(path, as_grey)
            im = (im - np.amin(im) * 1.0) / (np.amax(im) - np.amin(im))
            return im

        # Load image
        image = imread(path, as_grey)

        if to_float:
            # Convert to floating point matrix
            image = image.astype(np.float32)

        return image

    @staticmethod
    def sticht_images_vignete(images_and_passpoints, mode=StitchMode.MODE_MULTIBAND_BLENDING):
        weight_left_img = Img.calculate_weight(images_and_passpoints[0].image)
        weight_mask_left_img = DistortionCorrection.distortion_correction(images_and_passpoints[0].passpoints, weight_left_img)

        left_retificated_image = DistortionCorrection.distortion_correction(images_and_passpoints[0].passpoints,
                                                           images_and_passpoints[0].image)

        if len(images_and_passpoints) > 1:
            for i in range(1, len(images_and_passpoints)):
                weight_mask_right_img = DistortionCorrection.distortion_correction(images_and_passpoints[i].passpoints,
                                                                             Img.calculate_weight(images_and_passpoints[i].image))
                right_retificated_img = DistortionCorrection.distortion_correction(images_and_passpoints[i].passpoints,
                                                                                 images_and_passpoints[i].image)

                left_retificated_image, weight_mask_left_img = Img.stitch_two_pics(left_retificated_image, weight_mask_left_img,
                                                                         right_retificated_img, weight_mask_right_img,
                                                                         images_and_passpoints[i].passpoints, mode)
            return left_retificated_image
        else:
            return left_retificated_image


    @staticmethod
    def stitch_two_pics(retificated_img_1, weight_1_mask, retificated_img_2, weight_2_mask, passpoints_2, mode=StitchMode.MODE_STRONGEST):

        height1 = retificated_img_1.shape[0]
        width1 = retificated_img_1.shape[1]
        height2 = retificated_img_2.shape[0]
        width2 = retificated_img_2.shape[1]

        new_height = max(height1, height2)
        new_width = DistortionCorrectionPoint.get_max_distance(passpoints_2)[0]

        print new_width, ", ", new_height
        stitched_image = np.empty((new_height, new_width, 3))
        new_weight = np.empty((new_height, new_width, 3))

        vec3_zero = np.array([0, 0, 0])

        if mode == StitchMode.MODE_MULTIBAND_BLENDING:
            sig1_img1 = np.std(retificated_img_1[:, :, 0])
            sig2_img1 = np.std(retificated_img_1[:, :, 1])
            sig3_img1 = np.std(retificated_img_1[:, :, 2])

            sig1_img2 = np.std(retificated_img_2[:, :, 0])
            sig2_img2 = np.std(retificated_img_2[:, :, 1])
            sig3_img2 = np.std(retificated_img_2[:, :, 2])

            sigma_1_all = [sig1_img1, sig2_img1, sig3_img1]
            sigma_2_all = [sig1_img2, sig2_img2, sig3_img2]

            low_pass_retificated_img_1 = gaussian_filter(retificated_img_1, sigma_1_all)
            high_pass_retificated_img_1 = np.subtract(retificated_img_1, low_pass_retificated_img_1)

            low_pass_retificated_img_2 = gaussian_filter(retificated_img_2, sigma_2_all)
            high_pass_retificated_img_2 = np.subtract(retificated_img_2, low_pass_retificated_img_2)

        for y in xrange(new_height):
            for x in xrange(new_width):

                if x >= width1 or y >= height1:
                    point_weight1 = 0
                    point_color1 = vec3_zero
                else:
                    point_weight1 = weight_1_mask[y, x, 0]
                    point_color1 = retificated_img_1[y, x, :]

                    if mode == StitchMode.MODE_MULTIBAND_BLENDING:
                        point_low_pass1 = low_pass_retificated_img_1[y, x, :]
                        point_high_pass1 = high_pass_retificated_img_1[y, x, :]

                if x >= width2 or y >= height2:
                    point_weight2 = 0
                    point_color2 = vec3_zero
                else:
                    point_weight2 = weight_2_mask[y, x, 0]
                    point_color2 = retificated_img_2[y, x, :]

                    if mode == StitchMode.MODE_MULTIBAND_BLENDING:
                        point_low_pass2 = low_pass_retificated_img_2[y, x, :]
                        point_high_pass2 = high_pass_retificated_img_2[y, x, :]

                if mode == StitchMode.MODE_STRONGEST:
                    if point_weight1 > point_weight2:
                        stitched_image[y, x, :] = retificated_img_1[y, x, :]
                        new_weight[y, x] = point_weight1
                    else:
                        stitched_image[y, x, :] = retificated_img_2[y, x, :]
                        new_weight[y, x] = point_weight2
                elif mode == StitchMode.MODE_SUM:
                    total_weight = point_weight1 + point_weight2
                    if total_weight != 0:
                        stitched_image[y, x:] = (point_weight1 * point_color1 + point_weight2 * point_color2) / total_weight
                    new_weight[y, x] = total_weight / 2.0
                elif mode == StitchMode.MODE_MULTIBAND_BLENDING:
                    total_weight = point_weight1 + point_weight2
                    value = (point_weight1 * point_low_pass1 + point_weight2 * point_low_pass2) / total_weight

                    if point_weight1 > point_weight2:
                        value = value + point_high_pass1
                    else:
                        value = value + point_high_pass2

                    new_weight[y, x] = total_weight / 2.0
                    stitched_image[y, x, :] = value

        return stitched_image, new_weight

    @staticmethod
    def calculate_weight(img):
        dim_y, dim_x = (img.shape[0], img.shape[1])

        x_left = np.linspace(0, 1, dim_x / 2)
        x_right = x_left[::-1]
        x = np.concatenate((x_left, x_right))

        y_upper = np.transpose(np.linspace(0, 1, dim_y / 2))
        y_lower = y_upper[::-1]
        y = np.concatenate((y_upper, y_lower))

        weight = np.outer(x, y)

        weight = np.transpose(weight)

        weight3d = np.empty((weight.shape[0], weight.shape[1], 3))

        weight3d[:, :, 0] = weight
        weight3d[:, :, 1] = weight
        weight3d[:, :, 2] = weight

        #imsave('../gGebaeude/weight3d_old.jpg', weight3d)

        return weight3d

    @staticmethod
    def calculate_weight_new(img):
        dim_y, dim_x = (img.shape[0], img.shape[1])

        center_x, center_y = dim_x / 2.0, dim_y / 2.0
        radius = max([center_x, center_y])
        weight_per_pixel = 1.0 / radius

        grid = np.meshgrid(np.arange(dim_x), np.arange(dim_y))
        coordinates = np.vstack(grid).reshape(2, -1).T

        weight3d = np.empty((dim_y, dim_x, 3))

        print "coordinates.shape: ", coordinates.shape
        print "weight3d.shape: ", weight3d.shape

        for x, y in coordinates:
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            weight = (radius - distance) * weight_per_pixel
            # Der Radius kann kleiner sein als die gemessene Distanz, weil der Radius min(dim_x, dim_y) ist.
            # Das hat zur Folge, dass negative Gewichte rauskommen können, was ausgeglichen werden muss.
            #weight = max(0, weight)

            weight3d[y, x, :] = weight

        imsave('../gGebaeude/weight3d_new.jpg', weight3d)
        return weight3d

    @staticmethod
    def get_2d_rotation_matrix(rad):
        rotation_matrix = np.zeros((2, 2))

        rotation_matrix[0, 0] = math.cos(rad)
        rotation_matrix[0, 1] = -math.sin(rad)

        rotation_matrix[1, 0] = math.sin(rad)
        rotation_matrix[1, 1] = math.cos(rad)

        return rotation_matrix

    @staticmethod
    def get_2d_scale_matrix(scale):
        scale_matrix = np.zeros((2, 2))

        scale_matrix[0, 0] = scale
        scale_matrix[1, 1] = scale

        return scale_matrix

    @staticmethod
    def get_2d_x_scale_matrix(scale):
        x_scale_matrix = np.zeros((2, 2))

        x_scale_matrix[0, 0] = scale
        x_scale_matrix[1, 1] = 1

        return x_scale_matrix

    @staticmethod
    def get_2d_x_y_scale_matrix(x_scale, y_scale):
        x_scale_matrix = np.zeros((2, 2))

        x_scale_matrix[0, 0] = x_scale
        x_scale_matrix[1, 1] = y_scale

        return x_scale_matrix

    @staticmethod
    def get_x_3d_rotation_matrix(degrees):
        """Rotation through x axis"""

        rotation_matrix = np.zeros((3, 3))

        rotation_matrix[0, 0, 0] = 1

        rotation_matrix[1, 1, 1] = math.cos(degrees)
        rotation_matrix[1, 1, 2] = -math.sin(degrees)

        rotation_matrix[2, 2, 1] = math.sin(degrees)
        rotation_matrix[2, 2, 2] = math.cos(degrees)

        return rotation_matrix

    @staticmethod
    def get_scale_diagonal_matrix(scale_diag):
        scale_diagonal_matrix = np.zeros((2, 2))

        scale_diagonal_matrix[0, 0] = 1
        scale_diagonal_matrix[0, 1] = scale_diag
        scale_diagonal_matrix[1, 0] = 1
        scale_diagonal_matrix[1, 1] = 1
        return scale_diagonal_matrix

    @staticmethod
    def get_scale_orthogonal_matrix(scale_orthogonal):
        scale_orthogonal_matrix = np.zeros((2, 2))

        scale_orthogonal_matrix[0, 0] = 1
        scale_orthogonal_matrix[0, 1] = 1
        scale_orthogonal_matrix[1, 0] = scale_orthogonal
        scale_orthogonal_matrix[1, 1] = 1
        return scale_orthogonal_matrix


class Transform:
    @staticmethod
    def translate(matrix, trans_vector):
        return matrix + trans_vector


class RestructuringMethod(object):
    NearestNeighbor = 1
    BilinearInterpolation = 2

    @staticmethod
    def affine_transform(image,
                         transform_matrix,
                         translation_vector,
                         restructuring_method=BilinearInterpolation):
        from numpy.linalg import inv

        new_x_size = int(image.shape[0] * 1.5)
        new_y_size = int(image.shape[1] * 1.5)

        new_image = np.zeros((new_x_size, new_y_size, 3))

        # Get the inverse matrix for indirect restructuring
        trans_inv = inv(transform_matrix)

        for x in range(new_x_size):
            for y in range(new_y_size):
                new_coordinates = np.array([x, y])

                # First reverse translation
                new_coordinates = new_coordinates - translation_vector + np.array(
                    [0, -image.shape[1] / 2])  # -image.shape[0]/2

                # Reverse transformation
                new_coordinates = np.dot(new_coordinates, trans_inv)

                new_x = new_coordinates[0]
                new_y = new_coordinates[1]

                if restructuring_method == RestructuringMethod.NearestNeighbor:
                    new_x, new_y = RestructuringMethod.nearest_neighboor(new_x, new_y)

                if new_x > 0 and new_y > 0 and new_x < image.shape[0] and new_y < image.shape[1]:
                    if restructuring_method == RestructuringMethod.BilinearInterpolation:
                        new_image[x, y, :] = RestructuringMethod.bilinear_interpolation(image[:, :, :], new_x, new_y)
                    else:
                        new_image[x, y, :] = image[new_x, new_y, :]

        # back casting to uint8
        return new_image.astype(np.uint8)

    @staticmethod
    def bilinear_interpolation(image, x, y):
        x_left = int(x)
        x_right = int(x + 1)

        y_upper = int(y)
        y_lower = int(y + 1)

        # Because we added 1 on x and y, we could possibly be over
        # the range of the image
        image_x_max_index = image.shape[0] - 1
        image_y_max_index = image.shape[1] - 1

        if x_right > image_x_max_index or y_lower > image_y_max_index:
            return image[x, y]

        # calculate areas
        a1 = (x - x_left) * (y - y_upper)
        a2 = (x_right - x) * (y - y_upper)
        a3 = (x - x_left) * (y_lower - y)
        a4 = (x_right - x) * (y_lower - y)

        grey_value_left_upper = image[x_left, y_upper]
        grey_value_right_upper = image[x_right, y_upper]

        grey_value_left_lower = image[x_left, y_lower]
        grey_value_right_lower = image[x_right, y_lower]

        bilinear_interpolated_gray_value = grey_value_left_upper * a4 + grey_value_right_upper * a3 + \
                                           grey_value_left_lower * a2 + grey_value_right_lower * a1

        return bilinear_interpolated_gray_value

    @staticmethod
    def nearest_neighboor(x, y):
        # round coordinates
        new_x = int(x + 0.5)
        new_y = int(y + 0.5)

        return new_x, new_y


class DistortionCorrection(object):
    @staticmethod
    def generate_distort_correction_mat(points):

        equalisation_matrix = np.zeros(8)
        target_points = []

        for point in points:
            tmp_entry = [point.pass_point_x, point.pass_point_y, 1, 0, 0, 0, -point.target_point_x * point.pass_point_x,
                         -point.target_point_x * point.pass_point_y]
            equalisation_matrix = np.vstack((equalisation_matrix, tmp_entry))

            tmp_entry = [0, 0, 0, point.pass_point_x, point.pass_point_y, 1, -point.target_point_y * point.pass_point_x,
                         -point.target_point_y * point.pass_point_y]
            equalisation_matrix = np.vstack((equalisation_matrix, tmp_entry))
            target_points.append(point.target_point_x)
            target_points.append(point.target_point_y)

        # delete first pseudo entry
        equalisation_matrix = np.delete(equalisation_matrix, 0, 0)

        target_points = np.transpose(target_points)

        pseudo_inverse = np.linalg.pinv(equalisation_matrix)

        return pseudo_inverse.dot(target_points)

    @staticmethod
    def distortion_correction(points, image_orig, use_bilinear_interpolation=True):

        if DEBUG:
            return DistortionCorrection_speed.distortion_correction(points, image_orig)

        a = DistortionCorrection.generate_distort_correction_mat(points)
        max_x, max_y = DistortionCorrectionPoint.get_max_distance(points)
        new_image = np.ones((max_y, max_x, 3)) * -1

        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        b1 = a[3]
        b2 = a[4]
        b3 = a[5]
        c1 = a[6]
        c2 = a[7]

        for y in np.arange(0, new_image.shape[0]):
            for x in np.arange(0, new_image.shape[1]):
                denominator = ((b1 * c2 - b2 * c1) * x) + ((a2 * c1 - a1 * c2) * y) + (a1 * b2) - (a2 * b1)

                new_x = ((b2 - c2 * b3) * x + (a3 * c2 - a2) * y + a2 * b3 - a3 * b2) / denominator
                new_y = ((b3 * c1 - b1) * x + (a1 - a3 * c1) * y + a3 * b1 - a1 * b3) / denominator

                if new_x > 0 and new_y > 0 and new_x < image_orig.shape[1] and new_y < image_orig.shape[0]:
                    if use_bilinear_interpolation:
                        new_image[y, x, :] = RestructuringMethod.bilinear_interpolation(image_orig[:, :, :], new_y,
                                                                                        new_x)
                    else:
                        new_image[y, x, :] = image_orig[new_y, new_x, :]

        return new_image


#import matplotlib.pyplot as plt

class DistortionCorrection_speed(object):
    @staticmethod
    def distortion_correction(points, image_orig):
        count_points = len(points)

        src = np.zeros((count_points, 2))
        dst = np.zeros((count_points, 2))

        for i in range(count_points):
            dst[i][0] = points[i].get_pass_point_x()
            dst[i][1] = points[i].get_pass_point_y()

            src[i][0] = points[i].get_target_point_x()
            src[i][1] = points[i].get_target_point_y()

        max_x, max_y = DistortionCorrectionPoint.get_max_distance(points)


        tform3 = tf.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = tf.warp(image_orig, tform3, output_shape=(max_y, max_x))

        if False:
            margins = dict(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
            text = image_orig

            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 3))
            fig.subplots_adjust(**margins)
            plt.gray()
            ax1.imshow(text)
            ax1.plot(dst[:, 0], dst[:, 1], '.r')
            ax1.axis('off')
            ax2.imshow(warped)
            ax2.axis('off')

            plt.show()

        return warped


class ImageAndPasspoints(object):
    def __init__(self, img, passpoints=[]):
        self.image = img
        self.passpoints = passpoints


class DistortionCorrectionPoint(object):
    @staticmethod
    def get_max_distance(points):
        tmp_x_max = -1
        tmp_y_max = -1

        for point in points:
            if tmp_x_max < point.get_target_point_x():
                tmp_x_max = point.get_target_point_x()
            if tmp_y_max < point.get_target_point_y():
                tmp_y_max = point.get_target_point_y()

        return tmp_x_max, tmp_y_max


    @staticmethod
    def set_move_to_right_in_array(points, right_pixel):
        for point in points:
            point.set_move_to_right(right_pixel)



    def __init__(self, pass_x, pass_y, target_x, target_y):
        self.pass_point_x = pass_x
        self.pass_point_y = pass_y
        self.target_point_x = target_x
        self.target_point_y = target_y

        self._move_to_right = 0

    def set_move_to_right(self, right_pixel):
        self._move_to_right = right_pixel

    def get_target_point_x(self):
        return self.target_point_x+self._move_to_right

    def get_target_point_y(self):
        return self.target_point_y

    def get_pass_point_x(self):
        return self.pass_point_x

    def get_pass_point_y(self):
        return self.pass_point_y


class Signal(object):

    @staticmethod
    def make_sequence(dim_t, dim_y, dim_x, v):
        wavelength = 0.5 * dim_x
        frequency = v / wavelength

        rad_per_pixel = np.pi / dim_x

        signal = np.empty((dim_x))

        for x in range(dim_x):
            pos_in_rad = rad_per_pixel * x
            value = np.sin(pos_in_rad)
            signal[x] = value

