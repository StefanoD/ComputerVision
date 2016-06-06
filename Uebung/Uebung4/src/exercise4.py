# -*- coding: utf-8 -*-

from scipy.io import loadmat
import matplotlib.pyplot as plt

from libcore import Signal
from libcore import Dia
import numpy as np


def aufgabe_1b():
    import time

    start = time.time()
    sinus_sequence = Signal.make_sequence(50, 10, 50, 1)
    end = time.time()

    print "make_sequence: ", end - start

    start = time.time()
    sinus_sequence = Signal.make_sequence_2(50, 10, 50, 1)
    end = time.time()

    print "mean: ", np.mean(sinus_sequence)
    normed = sinus_sequence + abs(np.min(sinus_sequence))
    contrast = (np.max(normed) - np.min(normed)) / (np.max(normed) + np.min(normed))
    print "contrast: ", contrast
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

def aufgabe_3b(v_parameter):
    sinus_sequence = Signal.make_sequence(dim_t=100, dim_y=5, dim_x=50, v=v_parameter)
    signal_left, signal_right = Signal.bug(sinus_sequence)
    correlation = Signal.detector(signal_left, signal_right, tau=1.1)

    # Korrelation ist am hoechsten, bei v*2*PI.
    # 2*PI ist in unserem Beispiel 25 Pixel. Also bei v * 25.

    print "correlation links-rechts: ", correlation

    return correlation


def aufgabe_3c(v_parameter):
    sinus_sequence = Signal.make_sequence(dim_t=100, dim_y=5, dim_x=50, v=v_parameter)
    signal_left, signal_right = Signal.bug(sinus_sequence)
    correlation = Signal.detector(signal_right, signal_left, tau=1.1)

    # Korrelation ist am hoechsten, bei v*2*PI.
    # 2*PI ist in unserem Beispiel 25 Pixel. Also bei v * 25.

    print "correlation rechts-links: ", correlation

    return correlation


def aufgabe_3d():
    v = 5
    corr_left = aufgabe_3b(v)
    corr_right = aufgabe_3c(v)

    corr_total = corr_right - corr_left

    print "corr_total: ", corr_total


def main():
    aufgabe_1b()


if __name__ == "__main__": main()
