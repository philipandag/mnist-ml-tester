from tones import tones
from tones import tones_to_pitch

#tworzenie listy nut występujących w każdym z segmentów czasu
def get_notes(times, frequencies, spectrogram, spectrogram_min_volume, min__frequency=1000, max_frequency=8000):
    notes = [[] for _ in times]
    for time_index in range(len(times)):
        for frequency_index in range(len(frequencies)):
            if max_frequency <= frequencies[frequency_index] <= min__frequency:
                continue

            if spectrogram[frequency_index][time_index] < spectrogram_min_volume:
                continue
        
            best_fit_frequency = -1
            lowest_difference = 10000000
            for tone in list(tones.keys()):
                if abs(tone - frequencies[frequency_index]) < lowest_difference:
                    lowest_difference = abs(tone - frequencies[frequency_index])
                    best_fit_frequency = tone

            notes[time_index].append( (best_fit_frequency, spectrogram[frequency_index][time_index]) )

    return notes
#skalowanie wartosci gloscnosci 0-100 dla zadanej wartosci maksymalnej
def scale_volumes_to_percent(notes, value_for_100_percent):
    for t in range(len(notes)):
        for n in range(len(notes[t])):
            notes[t][n] = (notes[t][n][0], notes[t][n][1]/value_for_100_percent*100)

#laczenie duplikatow dzwiekow sumujac ich glosnosc
def sum_duplicates(notes):
    to_delete = [[] for _ in range(len(notes))]
    for time in range(len(notes)):
        if len(notes[time]) < 2:
            continue

        for i in range(len(notes[time])):
            for j in range(i+1, len(notes[time])):
                if notes[time][i][0] == notes[time][j][0]:
                    notes[time][i] = (notes[time][i][0], notes[time][i][1] + notes[time][j][1])
                    to_delete[time].append(j)

    for time in range(len(to_delete)):
        to_delete[time].sort(reverse=True) # should be sorted to avoid changing indexes
        for i in to_delete[time]:
            notes[time].pop(i)


def max_value(notes):
    m = 0
    for time in notes:
        for note in time:
            if note[1] > m:
                m = note[1]
    return m

#wypisanie listy nut
def print_notes_list(notes, times):
    for i in range(len(notes)):
        print("\nTime ", times[i], "s : ")
        for note in notes[i]:
            print("\tf:", tones[note[0]], "strength:", note[1], "%")

#sortowanie listy nut z malejaca intensywnoscia
def sort_by_intensity(notes):
    for time in notes:
        time.sort(key=lambda a: a[1], reverse=True)
