# -*- coding: utf-8 -*-

from scipy.io import loadmat
import matplotlib.pyplot as plt

from libcore import Signal
import numpy as np


class Dia(object):
    def __init__(self, sequenz_image):

        self.figure, self.subplot = plt.subplots()
        plt.gray()
        # Key Listner
        self.figure.canvas.mpl_connect('key_press_event', self._key_listner)

        self.current_frame = 0

        self.sequenz_iamges = sequenz_image
        self.max_sequenz_pics = sequenz_image.shape[0]

    def _redraw_image(self):
        self.subplot.imshow(self.sequenz_iamges[self.current_frame, :, :])
        plt.draw()

    def show_seq(self, start_frame=0):
        if start_frame != 0:
            self.current_frame = start_frame
        else:
            self.current_frame = 0

        self._redraw_image()
        plt.show()

    def _key_listner(self, key_event):
        if key_event.key == 'w':
            self.current_frame += 1
            if self.current_frame < self.max_sequenz_pics:
                self._redraw_image()

        elif key_event.key == 'q':
            plt.close()

def aufgabe_1b():
    import time

    start = time.time()
    sinus_sequence = Signal.make_sequence(50, 10, 50, 1)
    end = time.time()

    print "make_sequence: ", end - start

    start = time.time()
    sinus_sequence = Signal.make_sequence_2(50, 10, 50, 1)
    end = time.time()

    print "make_sequence_2: ", end - start

    Dia(sinus_sequence).show_seq()

def aufgabe_1a():
    robot_corridor_mat = loadmat('../robot-corridor.mat')
    robot_corridor_sequenz = robot_corridor_mat['seq']

    roboter_corrdor = Dia(robot_corridor_sequenz)
    roboter_corrdor.show_seq()

def aufgabe_2():
    signals = loadmat('../signals.mat')

    step = signals['step'][0]
    rectangle = signals['rectangle'][0]

    tau = 2.0

    lps = Signal.lowpass(step, tau)

    plt.plot(np.arange(len(step)), step)

    plt.plot(np.arange(len(lps)), lps)
    plt.show()

    lps = Signal.lowpass(rectangle, tau)

    plt.plot(np.arange(len(rectangle)), rectangle)
    plt.plot(np.arange(len(lps)), lps)
    plt.show()

def aufgabe_3a():
    signals = loadmat('../signals.mat')
    step = signals['step'][0]
    rectangle = signals['rectangle'][0]

    tau = 2.0

    correlation = Signal.detector(rectangle, rectangle, tau)

def aufgabe_3b():
    sinus_sequence = Signal.make_sequence(dim_t=100, dim_y=5, dim_x=50, v=35)

    eye_left_pos_x, eye_left_pos_y = (10, 1)
    eye_right_pos_x, eye_right_pos_y = (20, 1)

    signal_left = [sinus_signal[eye_left_pos_y, eye_left_pos_x] for sinus_signal in sinus_sequence]
    signal_right = [sinus_signal[eye_right_pos_y, eye_right_pos_x] for sinus_signal in sinus_sequence]

    correlation = Signal.detector(signal_left, signal_right, tau=1.1)

    # Korrelation ist am hoechsten, bei v*2*PI.
    # 2*PI ist in unserem Beispiel 25 Pixel. Also bei v * 25.

    print "correlation: ", correlation

def main():
    aufgabe_3b()



def test():
    print "test"


if __name__ == "__main__": main()
