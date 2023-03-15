import matplotlib
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np


def cuttop(spectrogram, n):
    for i in range(len(spectrogram)):
        sorted = np.array(spectrogram[i])
        sorted.sort()
        maxn = sorted[-n]
        for j in range(len(spectrogram[i])):
            if spectrogram[i][j] < maxn:
                spectrogram[i][j] = 0


    return spectrogram

                

def cut(spectrogram, mid):
    for i in range(len(spectrogram)):
        for j in range(len(spectrogram[i])):
            if spectrogram[i][j] < mid:
                spectrogram[i][j] = 0

    return spectrogram

def print_spectrogram(spectrogram):
    for time in range(len(times)):
        print("\n\nt", time, ": ")
        for frequency in range(len(spectrogram[time])):
            print("t: ", time, "f: ", frequency, ": ", spectrogram[time][frequency], "%")



sample_rate, samples = wavfile.read('piano.wav')
#usrednione probki fali - jeden kanal
samples = np.mean(np.array(samples), axis=1) 

frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=4096)

mid = np.mean(spectrogram)
cut(spectrogram, mid)




plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.ylim(0, 2000)
plt.show()