import numpy as np
import scipy.io
from scipy.misc import imread, imsave, toimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class correlationDetector(object):
    """docstring for correlationDetector"""
    def __init__(self):
        super(correlationDetector, self).__init__()

        tau = 1.3

        # mat = scipy.io.loadmat('robot-corridor.mat')
        # images = mat['seq']
        # self.show_seq(images)
        # sequence = self.make_seq(100,50,5,5)
        # self.show_seq(sequence)
        mat = scipy.io.loadmat('signals.mat')
        step = mat['step']
        lpsignal = self.lowpass(step[0], tau, True)
        rectangle = mat['rectangle']
        lpsignal = self.lowpass(rectangle[0], tau, True)
        # exit()

        tau = 1.1

        results = np.empty(51)
        print(results.shape)
        for i in range(-25,26,1):
            sequence = self.make_seq(100,50,5,i)
            # print(i+25)
            results[i+25] = self.detector(sequence[:,0,10], sequence[:,0,20], tau)

        lResults = np.empty(51)
        for i in range(-25,26,1):
            sequence = self.make_seq(100,50,5,i)
            # print(i+25)
            lResults[i+25] = self.lDetector(sequence[:,0,10], sequence[:,0,20], tau)

        # print(results)
        # print(lResults)
        print(len(results))
        plt.plot(results)
        plt.plot(lResults)
        # diff = np.empty(51)
        # diff = lResults - results
        # print('diff', diff)
        # plt.plot(diff)
        # plt.show()

    def detector(self, signal1, signal2, tau):

        lpsignal = self.lowpass(signal1, tau)

        corr = lpsignal * signal2
        corrSum = corr.sum()

        return corrSum

    def lDetector(self, signal1, signal2, tau):

        lpsignal = self.lowpass(signal2, tau)

        corr = lpsignal * signal1
        corrSum = corr.sum()

        return corrSum

    def lowpass(self, signal, tau, plot=False):

        a = 1.0/tau
        s = np.empty(signal.shape, dtype=np.float64)

        for n in range(0, len(signal)):
            if n == 0:
                s[n] = 0.0
            else:
                s[n] = a*signal[n] + (1.0 - a)*s[n-1]

        if plot:
            plt.plot(s)
            plt.plot(signal)
            plt.show()

        return s


    def show_seq(self, images):
        
        self.i = 0
        self.images = images

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.images[self.i,:,:], cmap = cm.Greys_r)
        # fig.figimage(images[0,:,:], cmap = cm.Greys_r)
        self.fig.canvas.mpl_connect('key_press_event', self.key_event)
        plt.show()


    def key_event(self, e):
        print (e.key)
        self.i += 1
        if self.i < self.images.shape[0]:
            self.ax.imshow(self.images[self.i,:,:], cmap = cm.Greys_r)
            plt.draw()


    def make_seq(self, dimT, dimX, dimY, v):

        img = np.zeros((dimT, dimY, dimX))
        wavelength = 0.5*dimX

        for i in range(1, dimT + 1):
            for x in range(dimX):
                img[i - 1,:,x] = 0.5 * np.sin((2*np.pi/wavelength)*(x - ((i-1)*v)))

        # self.show_seq(img)


        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)
        # self.ax.imshow(img[0,:,:], cmap = cm.Greys_r)
        # plt.show()

        # print(img)

        return img


if __name__ == '__main__':
    correlationDetector()