import matplotlib
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
from tones import tones


#zero any values other than top n in each time segment
def cut_top_n(spectrogram, n):
    for i in range(len(spectrogram)):
        sorted = np.array(spectrogram[i])
        sorted.sort()
        maxn = sorted[-n]
        for j in range(len(spectrogram[i])):
            if spectrogram[i][j] < maxn:
                spectrogram[i][j] = 0


    return spectrogram

                
#zero any values lower than mid
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

frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=8162)

plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.ylim(0, 2000)

for frequency in list(tones.keys()):
    plt.plot(times, [frequency for _ in range(len(times))], "r", linewidth="0.1")

plt.show()