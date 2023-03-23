class Tone:
    name=""
    index=-1
    value = -1
    
    def __init__(self, name, index, value = -1):
        self.name = name
        self.index = index
        self.value = value


toneNames = {
    16.35 : "C",
    17.32 : "C#",
    18.35 : "D",
    19.45 : "D#",
    20.60 : "E",
    21.83 : "F",
    23.12 : "F#",
    24.50 : "G",
    25.96 : "G#",
    27.50 : "A",
    29.14 : "A#",
    30.87 : "B",

    32.70 : "C",
    34.65 : "C#",
    36.71 : "D",
    38.89 : "D#",
    41.20 : "E",
    43.65 : "F",
    46.25 : "F#",
    49.00 : "G",
    51.91 : "G#",
    55.00 : "A",
    58.27 : "A#",
    61.74 : "B",

    65.41 : "C",
    69.30 : "C#",
    73.42 : "D",
    77.78 : "D#",
    82.41 : "E",
    87.31 : "F",
    92.50 : "F#",
    98.00 : "G",
    103.83 : "G#",
    110.00 : "A",
    116.54 : "A#",
    123.47 : "B",

    130.81 : "C",
    138.59 : "C#",
    146.83 : "D",
    155.56 : "D#",
    164.81 : "E",
    174.61 : "F",
    185.00 : "F#",
    196.00 : "G",
    207.65 : "G#",
    220.00 : "A",
    233.08 : "A#",
    246.94 : "B",

    261.63 : "C",
    277.18 : "C#",
    293.66 : "D",
    311.13 : "D#",
    329.63 : "E",
    349.23 : "F",
    369.99 : "F#",
    392.00 : "G",
    415.30 : "G#",
    440.00 : "A",
    466.16 : "A#",
    493.88 : "B",

    523.25 : "C",
    554.37 : "C#",
    587.33 : "D",
    622.25 : "D#",
    659.25 : "E",
    698.46 : "F",
    739.99 : "F#",
    783.99 : "G",
    830.61 : "G#",
    880.00 : "A",
    932.33 : "A#",
    987.77 : "B",

    1046.50 : "C",
    1108.73 : "C#",
    1174.66 : "D",
    1244.51 : "D#",
    1318.51 : "E",
    1396.91 : "F",
    1479.98 : "F#",
    1567.98 : "G",
    1661.22 : "G#",
    1760.00 : "A",
    1864.66 : "A#",
    1975.53 : "B",

    2093.00 : "C",
    2217.46 : "C#",
    2349.32 : "D",
    2489.02 : "D#",
    2637.02 : "E",
    2793.83 : "F",
    2959.96 : "F#",
    3135.96 : "G",
    3322.44 : "G#",
    3520.00 : "A",
    3729.31 : "A#",
    3951.07 : "B",
    
    4186.01 : "C",
    4439.92 : "C#",
    4698.63 : "D",
    4978.03 : "D#",
    5274.04 : "E",
    5587.65 : "F",
    5919.91 : "F#",
    6271.93 : "G",
    6644.88 : "G#",
    7040.00 : "A",
    7458.62 : "A#",
    7902.13 : "B",
}

universalTones = {
    
    "C" : Tone("C", 0,1),
    "C#" : Tone("C#", 1,1),
    "D" : Tone("D", 2,1),
    "D#" : Tone("D#", 3,1),
    "E" : Tone("E", 4,1),
    "F" : Tone("F", 5,1),
    "F#" :  Tone("F#", 6,1),
    "G" : Tone("G", 7,1),
    "G#" : Tone("G#", 8,1),
    "A" : Tone("A", 9,1),
    "A#" : Tone("A#", 10,1),
    "B" : Tone("B",11,1)
}

def findTone(freq):
    lastf = 16.35 #frequency of C0
    target = -1
    for f in toneNames.keys():
        if(f>freq):
            if(abs(f-freq)<abs(lastf-freq)):
                target = f
            else:
                target = lastf
            break
        else:
            lastf = f
    if target == -1:
        target = 7902.13 #frequency of B8
    
    return universalTones[toneNames[target]]

def findToneByName(name):
    return universalTones[name]


#def findInterval(tone1: Tone, tone2: Tone):
#    return abs(tone1.index - tone2.index)

def findInterval(tone1: Tone, tone2: Tone):
    if(tone1.index <= tone2.index):
        return tone2.index - tone1.index
    else:
        return (12-(tone1.index - tone2.index))

#testTone = findTone(32.70)
#testTone2 = findTone(30.87)
#print(testTone.name, testTone.index)
#print(testTone2.name, testTone2.index)

#print(findInterval(testTone2, testTone))











'''
CRUX SACRA SIT MIHI LUX 
NON DRACO SIT MIHI DUX
VADE RETRO SATANA 
NUNQUAM SUADE MIHI VANA
SUNT MALA QUAE LIBAS 
IPSE VENENA BIBAS 
'''