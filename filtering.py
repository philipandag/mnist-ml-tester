
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
