# -*- coding: utf-8 -*-

from scipy.io import loadmat
import matplotlib.pyplot as plt

from libcore import Img


class Gui(object):

    def __init__(self):

        self.figure, self.subplot = plt.subplots( figsize=(10, 10))
        plt.gray()
        # Key Listner
        self.figure.canvas.mpl_connect('key_press_event', self.key_listner)

        self.current_frame = 0

        self.max_sequenz_pics = 0

    def show(self):
        plt.show()

    def show_seq(self, sequenz , start_frame = 0):



        if start_frame != 0:
            self.current_frame = start_frame
        else:
            self.current_frame = 0

        self.subplot.imshow(sequenz[self.current_frame,:,:])

        plt.show()

    def key_listner(self, key_event):
        if key_event.key == 'w':
            print "Pressed Weiter"
            self.current_frame += 1
            #if self.current_frame <

        elif key_event.key == 'q':
            plt.close()

def main():

    robot_corridor_mat = loadmat('../robot-corridor.mat')
    robot_corridor_sequenz = robot_corridor_mat['seq']

    display_gui = Gui()

    display_gui.show_seq(robot_corridor_sequenz)

    #




def test():
    print "test"



if __name__ == "__main__": main()
