from midiutil.MidiFile import MIDIFile
import pygame.mixer
import matplotlib.pyplot as plt
from tones import tones
from tones import tones_to_pitch

class MidiFile:
    def __init__(self, times):
        #time_step = times[1] - times[0]
        #tempo = 1/time_step*60 # rate per second * 60s per minute
        tempo = 60
        self.file = MIDIFile(1)
        self.file.addTrackName(0, 0, "Track")
        self.file.addTempo(0, 0, tempo)
        self.fileName = None
    
    def addNote(self, pitch, start, length, volume):
        pitch = int(pitch)
        volume = int(volume)
        self.file.addNote(0, 0, pitch, start, length, volume)
    
    def write(self, fileName):
        self.fileName = fileName
        with open(fileName, "wb") as output_file:
            self.file.writeFile(output_file)
    
    def play(self):
        pygame.mixer.init()
        pygame.mixer.music.load(self.fileName, "midi")
        pygame.mixer.music.play()
    
    def add_notes_list(self, notes, times):
        for t in range(len(notes)):
            for note in notes[t]:
                self.addNote(tones_to_pitch[note[0]], times[t], 1, note[1]*2.55) # note[1] czyli glosnosc jest w skali 0-100, midi przyjmuje w skali 0-255



def print_spectrogram(frequencies, times, spectrogram):
    for time in range(len(times)):
        print("\n\nt", times[time], ": ")
        for frequency in range(len(spectrogram[time])):
            print("t: ", times[time], "f: ", frequency, ": ", spectrogram[time][frequency], "%")


def plot_spectrogram(frequencies, times, spectrogram):
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.ylim(0, 2000)

    for frequency_index in list(tones.keys()):
        plt.plot(times, [frequency_index for _ in range(len(times))], "r", linewidth="0.1")

    plt.show()

