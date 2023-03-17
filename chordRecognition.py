import universalTones
from universalTones import *

class ChordTemplate:
    name=""
    intervals=[]
    value=-1
    
    def __init__(self, name, intervals):
        self.name = name
        self.intervals = intervals
        self.value = len(intervals)
        
chordTemplates = [
    ChordTemplate("5", [0,7]),
    ChordTemplate("maj", [0,4,7]),
    ChordTemplate("min", [0,3,7]),
    ChordTemplate("maj7", [0,4,7,11]),
    ChordTemplate("min7", [0,3,7,11])
]

class Chord:
    root : Tone =None
    template : ChordTemplate =None
    name : str =""
    
    def __init__(self, root: Tone, template: ChordTemplate):
        self.root = root
        self.template = template
        self.name = root.name + template.name 
        
    def value(self):
        return self.template.value
    
    def __str__(self):
        return self.name
    

#jakakolwiek nieprawidłowość dyskwalifikuje akord (prosta wersja algorytmu)
#TODO: algorytm który uwzględnia w jakim stopniu akord jest podobny do wzorca
def applyTemplate(root: Tone, tones: Tone, template: ChordTemplate) -> bool:
    intervals  =[]
    for tone in tones:
        intervals.append(findInterval(root, tone))
    
    if (set(intervals)==set(template.intervals)):
        return True
    else:
        return False
        

def getChordFromTones(tones: Tone):
    bestChord : Chord = None
    for root in tones:
        #zakładamy, że każdy dźwięk może być prymą
        for template in chordTemplates:
            if(applyTemplate(root, tones, template)):
                if(bestChord == None or template.value > bestChord.value()):
                    bestChord = Chord(root, template)

    return bestChord

def getChordFromFreqs(freqs):
    tones = []
    for f in freqs:
        tones.append(findTone(f))
    return getChordFromTones(tones)

def getChordFromTonenames(names):
    tones = []
    for name in names:
        tones.append(findToneByName(name))
    return getChordFromTones(tones)

#print(getChordFromFreqs([82.41,130.81,24.50,7902.13]))
print(getChordFromTonenames(["G", "B", "D"]))