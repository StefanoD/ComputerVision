import numpy as np
import scipy.io
from scipy.misc import imread, imsave, toimage
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import math

def main():
    print("Aufgabe1")
    aufgabe1()
    print("Aufgabe2")
    aufgabe2()




def ttc(seq):
    plot_x = list(range(len(seq)-1))
    plot_y = np.zeros((len(seq)-1, 1))

    subplot_count = 191
    img_count = 0
    for i in range(len(seq)-1):
        pic1 = seq[i]
        pic2 = seq[i+1]
        imsave(('pic%i.jpg') % img_count, pic1)
        imsave(('pic%i.jpg') % (img_count+1), pic2)

        logpic1 = logmap(pic1, pic1.shape[0], pic1.shape[1])
        logpic2 = logmap(pic2, pic2.shape[0], pic2.shape[1])
        imsave(('logpic%i.jpg') % img_count, logpic1)
        imsave(('logpic%i.jpg') % (img_count+1), logpic2)

        img_count += 2
        # plt.subplot(subplot_count)
        # plt.imshow(logpic1, cmap=plt.gray())

        # plt.subplot(subplot_count+1)
        # plt.imshow(logpic2)
        # subplot_count += 2

        uy, ux = optical_flow(logpic1, logpic2, 0.01)

        absolute = np.sqrt(uy**2 + ux**2)
        anz_absolute = len(absolute)

        bins = 10

        hist, edges = np.histogram(absolute, bins)
        # print (hist, edges)
        # exit()
        summe = 0
        count = 0
        for j in reversed(range(len(hist))):
            summe = summe + hist[j]
            count += 1
            if summe >= anz_absolute * 0.1:
                break


        # print (edges)
        time_to_contact = 1 / edges[count]

        plot_y[i] = time_to_contact

        

        # print (time_to_contact)
    print ("x", plot_x)
    print ("y", plot_y)
    # plt.subplot(subplot_count)
    plt.figure()
    plt.plot(plot_x, plot_y)
    plt.show()



def logmap(pic, h, w):
    result = np.zeros((h, w))

    center_x = pic.shape[1] / 2
    center_y = pic.shape[0] / 2

    radius = np.sqrt(center_y**2 + center_x**2)

    for x in range(0,w):
        for y in range(0,h):
            r = radius**(x/w)
            phi = np.pi + ((2 * np.pi) / h) * y

            cart_y, cart_x = pol2cart(r, phi)
            cart_x = round(cart_x + center_x)
            cart_y = round(cart_y + center_y)

            if cart_x >= 0 and cart_x < pic.shape[1] and cart_y >= 0 and cart_y < pic.shape[0]:
                result[y, x] = pic[cart_y, cart_x]

    return result

def cartmap(pic, h, w):
    result = np.zeros((h, w))

    center_x = w / 2
    center_y = h / 2

    # print (center_x, center_y)

    radius = np.sqrt(center_y**2 + center_x**2)

    for y in range(1, h+1):
        for x in range(1, w+1):
            y_trans = y - center_y
            x_trans = x - center_x

            #print(y_trans, x_trans)

            r, phi = cart2pol(y_trans, x_trans)
            # if r <= 0:
            #     continue

            y_polar = np.ceil((phi + np.pi) * (pic.shape[0] / (2 * np.pi)))

            exponent = 1 / (pic.shape[1] - 1)
            nenner = exponent * np.log(radius)

            x_polar = np.ceil(np.log(r) / nenner)

            # if np.log(r) == 0:
            #     print( y, x, y_trans, x_trans, r, nenner, y_polar, x_polar)

            # if y_polar == pic.shape[0]:
                # print (y, x, exponent, nenner)
            #if x_polar >= 0 and x_polar < pic.shape[1] and y_polar >= 0 and y_polar < pic.shape[0]:
            if (not(x_polar > pic.shape[1] or y_polar > pic.shape[0] or x_polar < 1 or y_polar < 1)):
                result[y-1, x-1] = pic[y_polar-1, x_polar-1]

    return result


def cart2pol(y, x):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return y, x

def aufgabe2():
    seq = scipy.io.loadmat('contact.mat')
    ttc(seq['seq'])

def aufgabe1():
    seq = scipy.io.loadmat('logtest.mat')
    pic = seq['pic']

    pic = pic*255

    logpic = logmap(pic, 128, 64)
    #imsave(('logmap_a1.jpg'), logpic)

    cartpic = cartmap(logpic, pic.shape[0], pic.shape[1])

    plt.subplot(131)
    plt.imshow(pic, cmap=plt.gray())
    plt.subplot(132)
    plt.imshow(logpic)
    plt.subplot(133)
    plt.imshow(cartpic)
    plt.show()



def optical_flow(pic1, pic2, theta):

    sigma = 1.5
    pic1 = pic1.astype(np.float64)
    pic2 = pic2.astype(np.float64)


    dt = pic2 - pic1

    flow_pic1 = pic1
    flow_pic2 = pic2



    filter_dx = np.array([[0.0,0.0,0.0], [0.5,0.0,-0.5], [0.0,0.0,0.0]])
    filter_dy = np.transpose(filter_dx)

    pic2_dx = ndi.convolve(flow_pic2,filter_dx)
    pic2_dy = ndi.convolve(flow_pic2,filter_dy)

    pic2_dxx = np.multiply(pic2_dx, pic2_dx)
    pic2_dyy = np.multiply(pic2_dy, pic2_dy)
    pic2_dxy = np.multiply(pic2_dx, pic2_dy)
    pic2_dxt = np.multiply(pic2_dx, dt)
    pic2_dyt = np.multiply(pic2_dy, dt)

    pic2_dxx = ndi.gaussian_filter(pic2_dxx, sigma)
    pic2_dyy = ndi.gaussian_filter(pic2_dyy, sigma)
    pic2_dxy = ndi.gaussian_filter(pic2_dxy, sigma)
    pic2_dxt = ndi.gaussian_filter(pic2_dxt, sigma)
    pic2_dyt = ndi.gaussian_filter(pic2_dyt, sigma)

    g = np.sqrt(pic2_dxx + pic2_dyy)



    ux = np.zeros(pic2.shape)
    uy = np.zeros(pic2.shape)
    for x in range(pic2.shape[1]):
        for y in range(pic2.shape[0]):

            A = np.array([[pic2_dxx[y,x], pic2_dxy[y,x]],[pic2_dxy[y,x],pic2_dyy[y,x]]])
            b = np.array([[pic2_dxt[y,x]],[pic2_dyt[y,x]]])

            A_eig = np.linalg.eig(A)[0]


            if A_eig[0] > theta and A_eig[1] > theta:

                u = -np.dot(np.linalg.inv(A), b)

                ux[y,x] = u[0]
                uy[y,x] = u[1]

            elif A_eig[1] < theta < A_eig[0] or A_eig[0] < theta < A_eig[1]:
                pixels = [(x-1, y), (x, y-1), (x+1, y), (x, y+1), (x-1, y-1), (x+1, y+1), (x-1, y+1), (x+1, y-1), (x, y)]
                A2 = np.zeros((1,9))
                b2 = np.zeros((1,9))
                for i in range(len(pixels)):
                    if pixels[i][1] >= 0 and pixels[i][1] < pic2.shape[0] and pixels[i][0] >= 0 and pixels[i][0] < pic2.shape[1]:
                        A2[0][i] = g[pixels[i][1],pixels[i][0]]
                        b2[0][i] = dt[pixels[i][1],pixels[i][0]]

                u_orth = np.dot(-np.linalg.pinv(np.dot(A2, A2.T)), (np.dot(A2, b2.T)))
                ux[y,x] = u_orth * pic2_dx[y,x] / g[y,x]
                uy[y,x] = u_orth * pic2_dy[y,x] / g[y,x]

    return uy, ux

if __name__ == '__main__':
    main()

