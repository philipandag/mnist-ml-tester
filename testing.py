from midiutil.MidiFile import MIDIFile
import pygame.mixer

class MidiFile:
    def __init__(self, tempo):
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


def print_spectrogram(spectrogram):
    for time in range(len(times)):
        print("\n\nt", time, ": ")
        for frequency in range(len(spectrogram[time])):
            print("t: ", time, "f: ", frequency, ": ", spectrogram[time][frequency], "%")

