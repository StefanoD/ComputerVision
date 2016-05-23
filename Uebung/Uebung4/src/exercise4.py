# -*- coding: utf-8 -*-

from scipy.io import loadmat
import matplotlib.pyplot as plt

from libcore import Signal


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
        self.subplot.imshow(self.sequenz_iamges[self.current_frame,:,:])
        plt.draw()

    def show_seq(self, start_frame = 0):
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

def main():

    robot_corridor_mat = loadmat('../robot-corridor.mat')
    robot_corridor_sequenz = robot_corridor_mat['seq']

    #roboter_corrdor = Dia(robot_corridor_sequenz)

    #roboter_corrdor.show_seq()

    sinus_sequence = Signal.make_sequence(10, 10, 50, 25)

    Dia(sinus_sequence).show_seq()




def test():
    print "test"



if __name__ == "__main__": main()
