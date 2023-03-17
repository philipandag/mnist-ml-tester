import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

import numpy as np
from tones import tones
from testing import *
from notes_list import *

sample_rate, samples = wavfile.read('piano.wav')

#usrednione probki fali - jeden kanal
samples = np.mean(np.array(samples), axis=1) 

frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=8162)

mid = np.mean(spectrogram)
max = np.max(spectrogram)


notes = get_notes(times, frequencies, spectrogram, max/10)
scale_volumes_to_percent(notes, max)    
sum_duplicates(notes)
sort_by_intensity(notes)

#print_notes_list(notes, times)

print_spectrogram(frequencies, times, spectrogram)
plot_spectrogram(frequencies, times, spectrogram)

file = MidiFile(times)
file.add_notes_list(notes, times)
file.write("test.midi")
