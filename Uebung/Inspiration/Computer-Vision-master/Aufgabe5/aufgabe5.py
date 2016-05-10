import numpy as np 
import scipy.io
from scipy.misc import imread, imsave, toimage
import scipy.ndimage as ndi
import matplotlib.pyplot as plt



def main():

    theta = 0.01

    flow = scipy.io.loadmat('flow.mat')
    flow_pic1 = flow['pic1']
    flow_pic2 = flow['pic2']
    toimage(flow_pic1).show()
    toimage(flow_pic2).show()
    optical_flow(flow_pic1, flow_pic2, theta)

    flow = scipy.io.loadmat('flowtest1.mat')
    flow_pic1 = flow['pic1']
    flow_pic2 = flow['pic2']
    optical_flow(flow_pic1, flow_pic2, theta)

    flow = scipy.io.loadmat('flowtest2.mat')
    flow_pic1 = flow['pic1']
    flow_pic2 = flow['pic2']
    optical_flow(flow_pic1, flow_pic2, theta)
    
    flow = scipy.io.loadmat('flowtest3.mat')
    flow_pic1 = flow['pic1']
    flow_pic2 = flow['pic2']
    optical_flow(flow_pic1, flow_pic2, theta)


def in_bounds(id):
        (y, x) = id
        return 0 <= x < width and 0 <= y < height

def optical_flow(pic1, pic2, theta):

    sigma = 1.5

    # pic1 = ndi.gaussian_filter(pic1, sigma)
    # pic2 = ndi.gaussian_filter(pic2, sigma)
    # pic1 = ndi.uniform_filter(pic1, 5, mode='reflect')
    # pic2 = ndi.uniform_filter(pic2, 5, mode='reflect')
    

    # pic1 = pic1.astype(np.float64)
    # pic2 = pic2.astype(np.float64)

    # pic1 = pic1.astype(np.bool_)
    # pic2 = pic2.astype(np.bool_)
    # pic1 = np.invert(pic1)
    # pic2 = np.invert(pic2)

    pic1 = pic1.astype(np.float64)
    pic2 = pic2.astype(np.float64)

    np.savetxt('pic1.txt', pic1, fmt='%1.2f')
    np.savetxt('pic2.txt', pic2, fmt='%1.2f')

    dt = pic2 - pic1
    # dt = dt.astype(np.int8)
    np.savetxt('dt.txt', dt.astype(np.int8), fmt='%1.2f')

    # print (abs(dt))
    toimage(abs(dt)).show()
    flow_pic1 = pic1
    flow_pic2 = pic2

    filter_dx = np.array([[0.0,0.0,0.0], [0.5,0.0,-0.5], [0.0,0.0,0.0]])
    filter_dy = np.transpose(filter_dx)

    # flow_pic1_grad_x = ndi.convolve(flow_pic1,filter_dx, output=np.float64, mode='nearest')
    # flow_pic1_grad_y = ndi.convolve(flow_pic1,filter_dy, output=np.float64, mode='nearest')

    pic2_dx = ndi.convolve(flow_pic2,filter_dx)
    pic2_dy = ndi.convolve(flow_pic2,filter_dy)

    # np.savetxt('pic2_dx.txt', pic2_dx, fmt='%1.2f')
    # np.savetxt('pic2_dy.txt', pic2_dy, fmt='%1.2f')

    pic2_dxx = np.multiply(pic2_dx, pic2_dx)
    pic2_dyy = np.multiply(pic2_dy, pic2_dy)
    pic2_dxy = np.multiply(pic2_dx, pic2_dy)
    pic2_dxt = np.multiply(pic2_dx, dt)
    pic2_dyt = np.multiply(pic2_dy, dt)

    # np.savetxt('pic2_dxx.txt', pic2_dxx, fmt='%1.2f')
    # np.savetxt('pic2_dyy.txt', pic2_dyy, fmt='%1.2f')
    # np.savetxt('pic2_dxy.txt', pic2_dxy, fmt='%1.2f')
    # np.savetxt('pic2_dxt.txt', pic2_dxt, fmt='%1.2f')
    # np.savetxt('pic2_dyt.txt', pic2_dyt, fmt='%1.2f')


    pic2_dxx = ndi.gaussian_filter(pic2_dxx, sigma)
    pic2_dyy = ndi.gaussian_filter(pic2_dyy, sigma)
    pic2_dxy = ndi.gaussian_filter(pic2_dxy, sigma)
    pic2_dxt = ndi.gaussian_filter(pic2_dxt, sigma)
    pic2_dyt = ndi.gaussian_filter(pic2_dyt, sigma)

    # np.savetxt('pic2_dxx_filter.txt', pic2_dxx, fmt='%1.2f')
    # np.savetxt('pic2_dyy_filter.txt', pic2_dyy, fmt='%1.2f')
    # np.savetxt('pic2_dxy_filter.txt', pic2_dxy, fmt='%1.2f')
    # np.savetxt('pic2_dxt_filter.txt', pic2_dxt, fmt='%1.2f')
    # np.savetxt('pic2_dyt_filter.txt', pic2_dyt, fmt='%1.2f')


    # filter_size = 3
    # pic2_dxx = ndi.uniform_filter(pic2_dxx, filter_size, mode='reflect')
    # pic2_dyy = ndi.uniform_filter(pic2_dyy, filter_size, mode='reflect')
    # pic2_dxy = ndi.uniform_filter(pic2_dxy, filter_size, mode='reflect')
    # pic2_dxt = ndi.uniform_filter(pic2_dxt, filter_size, mode='reflect')
    # pic2_dyt = ndi.uniform_filter(pic2_dyt, filter_size, mode='reflect')

    g = np.sqrt(pic2_dxx + pic2_dyy)


    ux = np.zeros(pic2.shape)
    uy = np.zeros(pic2.shape)
    for x in range(pic2.shape[1]):
        for y in range(pic2.shape[0]):



            A = np.array([[pic2_dxx[y,x], pic2_dxy[y,x]],[pic2_dxy[y,x],pic2_dyy[y,x]]])
            b = np.array([[pic2_dxt[y,x]],[pic2_dyt[y,x]]])

            # print("A", A.shape, A.dtype)
            # print("b", b.shape, b.dtype)

            # A = [[pic2_dxx[x,y], pic2_dxy[x,y]],[pic2_dxy[x,y],pic2_dyy[x,y]]]
            # b = [[pic2_dxt[x,y]],[pic2_dyt[x,y]]]

            A_eig = np.linalg.eig(A)[0]

            # print(A_eig[0],A_eig[1])

            if A_eig[0] > theta and A_eig[1] > theta:

                u = -np.dot(np.linalg.inv(A), b)

                # exit()
                ux[y,x] = u[0]
                uy[y,x] = u[1]

            elif A_eig[1] < theta < A_eig[0] or A_eig[0] < theta < A_eig[1]:
                pixels = [(x-1, y), (x, y-1), (x+1, y), (x, y+1), (x-1, y-1), (x+1, y+1), (x-1, y+1), (x+1, y-1), (x, y)] 
                A2 = np.zeros((1,9))
                b2 = np.zeros((1,9))
                for i in range(len(pixels)):
                    A2[0][i] = g[pixels[i][1],pixels[i][0]] #[pixelsic2_dx[pixels[1],pixels[0], pixelsic2_dy[pixels[1],pixels[0]]])#)
                    b2[0][i] = dt[pixels[i][1],pixels[i][0]]
                
                u_orth = np.dot(-np.linalg.pinv(np.dot(A2, A2.T)), (np.dot(A2, b2.T)))
                ux[y,x] = u_orth * pic2_dx[y,x] / g[y,x]
                uy[y,x] = u_orth * pic2_dy[y,x] / g[y,x]




    # toimage(flow_pic1_grad_x).show()
    # toimage(flow_pic1_grad_y).show()
    # toimage(ux).show()
    # toimage(uy).show()
    # toimage(np.subtract(pic1,pic2)*255).show()
    # toimage(pic1*255).show()
    # toimage(pic2*255).show()
    # print(pic1[30,20:30])
    # print(pic2[30,20:30])
    # print('max: ',ux.max(), uy.max())
    X,Y = np.meshgrid(np.arange(0,pic2.shape[0],1),np.arange(0,pic2.shape[1],1))
    print("shape", X.shape, Y.shape)
    # print(X,Y)
    print("max", ux.max(), uy.max())
    np.savetxt('ux.txt', ux, fmt='%1.2f')
    np.savetxt('uy.txt', uy, fmt='%1.2f')


    plt.quiver(X, Y, ux, -uy, units='xy', scale=1.0)
    imsave('ux.jpg', ux)
    imsave('uy.jpg', uy)
    # np.savetxt('pic1.txt', pic1.astype(int), delimiter=',')
    # np.savetxt('pic2.txt', pic2.astype(int), delimiter=',')
    plt.show()





if __name__ == '__main__':
    main()