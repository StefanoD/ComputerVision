#!/usr/bin/python3

import scipy.io
import numpy as np
import matplotlib.pyplot as plt


class ImgSeqProcessor:
    def __init__(self, img_seq):
        self.plt_cmap = plt.get_cmap('gray')
        self.img_seq = img_seq
        self.index = 1
        self.img_plot = plt.imshow(img_seq[0], cmap=self.plt_cmap)
        self.cid = self.img_plot.figure.canvas.mpl_connect('key_press_event', self.press)
        plt.show()

    def press(self, event):
        self.show_seq(self.img_seq)

    def show_seq(self, seq):
        if self.index < len(seq):
            print("Index: " + str(self.index))
            plt.imshow(seq[self.index], cmap=self.plt_cmap)
            self.index += 1
            plt.show()
        else:
            self.index = 0
            print("No more images.")
            # plt.close()


def logmap(img, h, w, use_mesh=False):
    res_img = np.zeros((h, w), dtype=img.dtype)
    if use_mesh:
        dx, dy = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        sx, sy = transform((dx, dy), res_img.shape)
        # do nearest neighbour
        sx, sy = sx.round().astype(int), sy.round().astype(int)
        # Mask for valid coordinates.
        mask = (0 <= sx) & (sx < img.shape[1]) & (0 <= sy) & (sy < img.shape[0])
        # Copy valid coordinates from source image
        res_img[dy[mask], dx[mask]] = img[sy[mask], sx[mask]]
    else:
        for y in np.arange(-res_img.shape[0] // 2, res_img.shape[0] // 2):
            for x in np.arange(0, res_img.shape[1]):
                #x = 63
                #y = res_img.shape[0]//2 #np.pi / 2  # res_img.shape[0] // 2
                src_pt = transform_cart2pol((x, y), img.shape, res_img.shape)
                # TODO x y
                s_x = src_pt[0] + (img.shape[1] // 2)
                s_y = src_pt[1] + (img.shape[0] // 2)
                # outer_circle = int(np.sqrt((img.shape[0] / 2) ** 2 + (img.shape[1] / 2) ** 2))
                # s_x = src_pt[0] + (outer_circle//2)
                # s_y = src_pt[1] + (outer_circle//2)

                s_x = np.clip(s_x, 0, img.shape[1] - 1)
                s_y = np.clip(s_y, 0, img.shape[0] - 1)
                res_img[y, x] = img[s_y, s_x]

    return res_img

def cartmap(polar_img, h, w):
    cart_img = np.zeros((h, w), dtype=polar_img.dtype)

    for x in range(-cart_img.shape[1]//2, cart_img.shape[1]//2):
        for y in range(-cart_img.shape[0]//2, cart_img.shape[0]//2):
            # x = 74#cart_img.shape[1] // 2
            # y = 0#cart_img.shape[0] // 2
            src_pt = transform_pol2cart((x,y), polar_img.shape, cart_img.shape)
            # print ("x=rho",src_pt[0], "y=phi", src_pt[1])
            s_x = src_pt[0] #+ cart_img.shape[1] // 2
            s_y = src_pt[1] + polar_img.shape[0] // 2#cart_img.shape[0] // 2
            # print ("sxy",s_x, s_y)
            s_x = np.clip(s_x, 0, polar_img.shape[1] - 1)
            s_y = np.clip(s_y, 0, polar_img.shape[0] - 1)
            # exit()

            cart_img[y + cart_img.shape[0]//2, x + cart_img.shape[1]//2] = polar_img[s_y, s_x]

    return cart_img

def transform_pol2cart(pt, polar_img_shape, cart_img_shape):
    x, y = pt
    # inner_circle = min(dst_shape) // 2
    # outer_circle = np.sqrt((dst_shape[0] / 2) ** 2 + (dst_shape[1] / 2) ** 2)

    rho, phi = cart2pol(x,y)


    inner_circle = min(cart_img_shape) // 2
    step_rho = polar_img_shape[1] / (inner_circle)
    step_phi = (polar_img_shape[0] // 2) / np.pi
    # print ("shape", (cart_img_shape[0] // 2), np.pi)
    # print ("inner circle",inner_circle)
    # print ("step_rho", step_rho, "step_phi", step_phi)

    return rho * step_rho, phi * step_phi


def cart2pol(x, y):
    # print("xy", x, y)
    rho = (np.sqrt(x ** 2 + y ** 2))
    # print("radius, ", np.sqrt(x ** 2 + y ** 2), rho)
    phi = np.arctan2(y, x)
    return rho, phi
















def transform_cart2pol(pt, src_shape, dst_shape):
    x, y = pt
    inner_circle = min(src_shape) // 2
    outer_circle = np.sqrt((src_shape[0] / 2) ** 2 + (src_shape[1] / 2) ** 2)
    
    step_rho = inner_circle // dst_shape[1]
    step_phi = np.pi / (dst_shape[0] // 2)
    
    return pol2cart(x * step_rho, y * step_phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def ttc(img1, img2):

    img1_polar = logmap(img1, 128, 64, use_mesh=False)
    img2_polar = logmap(img2, 128, 64, use_mesh=False)


    

    plt.subplot(131)
    plt.imshow(img1_polar, cmap=plt.gray())
    plt.subplot(132)
    plt.imshow(img2_polar)
    plt.show()








def img_to_float(img):
    return np.array(img).astype(float) / 255  # uint8 max


def exercise1():
    mat_content = scipy.io.loadmat('logtest.mat')
    img = img_to_float(mat_content['pic'])
    w = 64
    h = 128
    log_img = logmap(img, h, w, use_mesh=False)
    cart_img = cartmap(log_img, img.shape[0], img.shape[1])
    plt.subplot(131)
    plt.imshow(img, cmap=plt.gray())
    plt.subplot(132)
    plt.imshow(log_img)
    plt.subplot(133)
    plt.imshow(cart_img)
    plt.show()


def exercise2():
    mat_content = scipy.io.loadmat('contact.mat')
    seq = img_to_float(mat_content['seq'])
    # ImgSeqProcessor(seq)
    ttc(seq[0], seq[1])



def main():
    # exercise1()
    exercise2()


if __name__ == '__main__':
    main()
