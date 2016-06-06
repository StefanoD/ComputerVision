from libcore import *
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter, convolve
from scipy.misc import  toimage

def optical_flow(img1, img2, theta):
    print type(img1)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    x_mask_derivation = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
    y_mask_derivation = np.transpose(x_mask_derivation)

    Dx = convolve(img1, x_mask_derivation, mode='constant')
    Dy = convolve(img1, y_mask_derivation, mode='constant')

    Dt = img2 - img1


    #toimage(img1).show()
    #toimage(img2).show()
    #toimage(Dt).show()
    toimage(abs(Dt)).show()
    #exit()

    # Nichtlinearitaet
    Dx2 = Dx * Dx
    Dy2 = Dy * Dy
    Dxy = Dx * Dy
    Dxt = Dx * Dt
    Dyt = Dy * Dt

    sigma = 1.5
    GDx2 = gaussian_filter(Dx2, sigma)
    GDy2 = gaussian_filter(Dy2, sigma)
    GDxy = gaussian_filter(Dxy, sigma)
    GDxt = gaussian_filter(Dxt, sigma)
    GDyt = gaussian_filter(Dyt, sigma)

    u_value = np.zeros(img1.shape)
    v_value = np.zeros(img1.shape)


    for x in xrange(0, img1.shape[1]):
        for y in xrange(0, img1.shape[0]):
            A = np.array([[GDx2[y, x], GDxy[y, x]],
                          [GDxy[y, x], GDy2[y, x]]])

            b = np.array([GDxt[y, x], GDyt[y, x]])

            #Eigenwerte von A bestimmen, Folie 10 Kapitel 5
            eigenvalues_a, _ = np.linalg.eig(A)

            lambda_1 = eigenvalues_a[0]
            lambda_2 = eigenvalues_a[1]

            if lambda_1 > lambda_2 > theta:
                # Invertierung
                inverse = np.linalg.inv(A)
                #(u_x, u_y) = -inverse*b

                u= np.dot(-inverse,b)
                u_value[y,x] = u[0]
                v_value[y,x] = u[1]

                #exit(0)
            elif lambda_1 > theta > lambda_2:
                # Normalenfluss bestimmen
                pass
            else:
                print "moeh"

    plt.quiver(u_value, v_value)
    plt.show()


def get_images(mat_lab_img):
    img_array1 = loadmat(mat_lab_img)

    return img_array1['pic1'], img_array1['pic2']


def main():
    img1_1, img1_2 = get_images('../flowtest1.mat')
    Dia(np.array([img1_1, img1_2])).show_seq()

    optical_flow(img1_1, img1_2, 0.01)

    img2_1, img2_2 = get_images('../flowtest2.mat')
    Dia(np.array([img2_1, img2_2])).show_seq()

    optical_flow(img2_1, img2_2, 0.01)

    img3_1, img3_2 = get_images('../flowtest3.mat')
    Dia(np.array([img3_1, img3_2])).show_seq()

    optical_flow(img3_1, img3_2, 0.01)




if __name__ == "__main__": main()
