from libcore import *
from scipy.io import loadmat

def optical_flow(img1, img2, theta):
    pass


def get_images(mat_lab_img):
    img_array1 = loadmat(mat_lab_img)

    return img_array1['pic1'], img_array1['pic2']


def main():
    img1_1, img1_2 = get_images('../flowtest1.mat')
    Dia(np.array([img1_1, img1_2])).show_seq()

    img2_1, img2_2 = get_images('../flowtest2.mat')
    Dia(np.array([img2_1, img2_2])).show_seq()

    img3_1, img3_2 = get_images('../flowtest3.mat')
    Dia(np.array([img3_1, img3_2])).show_seq()


if __name__ == "__main__": main()