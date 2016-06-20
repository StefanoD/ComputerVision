from libcore import *
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter, convolve
from scipy.misc import  toimage

def optical_flow(img1, img2, theta):
    print type(img1)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    x_mask_derivation = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    y_mask_derivation = np.transpose(x_mask_derivation)

    Dx = convolve(img1, x_mask_derivation)#, mode='constant')
    Dy = convolve(img1, y_mask_derivation)#, mode='constant')

    Dt = img2 - img1

    #toimage(img1).show()
    #toimage(img2).show()
    #toimage(Dt).show()
    #toimage(abs(Dt)).show()
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
    GDt = gaussian_filter(Dt, sigma)

    u_value = np.zeros(img1.shape)
    v_value = np.zeros(img1.shape)

    Gg = np.sqrt(GDx2 + GDy2)

    for x in xrange(0, img1.shape[1]-1):
        for y in xrange(0, img1.shape[0]-1):
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

                u= np.dot(-inverse, b)

                u_value[y, x] = u[0]
                v_value[y, x] = u[1]
            elif lambda_2 < theta < lambda_1 or lambda_1 < theta < lambda_2:
                pixels = [(x-1, y), (x, y-1), (x+1, y), (x, y+1), (x-1, y-1), (x+1, y+1), (x-1, y+1), (x+1, y-1), (x, y)]
                m = np.zeros((1, 9))
                b = np.zeros(9)
                for index in range(len(pixels)):
                    m[0][index] = Gg[pixels[index][1], pixels[index][0]]
                    b[index] = GDt[pixels[index][1], pixels[index][0]]

                scalar_product = np.dot(m, m.T)

                if scalar_product == 0:
                    # Ganz grosse Zahl!
                    u_orth = 10000000
                else:
                    u_orth = np.dot(-m, b) / scalar_product

                if Gg[y, x] == 0:
                    u_value[y, x] = 0
                    v_value[y, x] = 0
                else:
                    # Dx und Dy ergeben den Gradient und Gg[y,x] ist die Laenge des Gradients.
                    # u_orth ist einfach nur ein Skalar, der die Geschwindigkeit an der Orthogonalen angibt.
                    u_value[y, x] = u_orth * Dx[y, x] / Gg[y, x]
                    v_value[y, x] = u_orth * Dy[y, x] / Gg[y, x]

    X, Y = np.meshgrid(np.arange(0, img2.shape[0], 1), np.arange(0, img2.shape[1], 1))

    plt.quiver(X, Y, u_value, -v_value, units='xy', scale=1.0)
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

    optical_flow(img3_1, img3_2, 0.03)




if __name__ == "__main__": main()
