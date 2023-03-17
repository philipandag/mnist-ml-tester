import matplotlib
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
from tones import tones


sample_rate, samples = wavfile.read('piano.wav')

#usrednione probki fali - jeden kanal
samples = np.mean(np.array(samples), axis=1) 

frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=8162)

mid = np.mean(spectrogram)
max = np.max(spectrogram)
mid = max/10

notes = [[] for _ in times]

for tone in list(tones.keys()):
    print(tone)

for time_index in range(len(times)):
    for frequency_index in range(len(frequencies)):
        if 8000 <= frequencies[frequency_index] <= 1000:
            continue

        if spectrogram[frequency_index][time_index] > mid:
            best_fit_frequency = -1
            lowest_difference = 10000000
            for tone in list(tones.keys()):
                if abs(tone - frequencies[frequency_index]) < lowest_difference:
                    lowest_difference = abs(tone - frequencies[frequency_index])
                    best_fit_frequency = tone

            notes[time_index].append( (tones[best_fit_frequency], spectrogram[frequency_index][time_index]/max) )
                
for time in notes:
    time.sort(key=lambda a: a[1], reverse=True)

for i in range(len(notes)):
    print("\nTime ", times[i], " : ")
    for note in notes[i]:
        print("\tf:", note[0], "strength:", note[1])

plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.ylim(0, 2000)

for frequency_index in list(tones.keys()):
    plt.plot(times, [frequency_index for _ in range(len(times))], "r", linewidth="0.1")

plt.show()