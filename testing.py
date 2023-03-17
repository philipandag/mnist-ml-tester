from midiutil.MidiFile import MIDIFile
import pygame.mixer

class MidiFile:
    def __init__(self, tempo):
        self.file = MIDIFile(1)
        self.file.addTrackName(0, 0, "Track")
        self.file.addTempo(0, 0, tempo)
        self.fileName = None
    
    def addNote(self, note, start, length, volume):
        note += 12 #our C0 has index 0 but midi C0 has index 12
        self.file.addNote(0, 0, note, start, length, volume)
    
    def write(self, fileName):
        self.fileName = fileName
        with open(fileName, "wb") as output_file:
            self.file.writeFile(output_file)
    
    def play(self):
        pygame.mixer.music.load(self.fileName, "midi")
        pygame.mixer.music.play()


def print_spectrogram(spectrogram):
    for time in range(len(times)):
        print("\n\nt", time, ": ")
        for frequency in range(len(spectrogram[time])):
            print("t: ", time, "f: ", frequency, ": ", spectrogram[time][frequency], "%")

