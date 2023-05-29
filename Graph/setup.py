import joblib
import numpy as np
from generate_graph import genGraph

mnist = joblib.load("mnist_784.joblib")

'''mnist = load_digits()
mnist.data = mnist.data * 16

mnist.data = 255 - mnist.data  # Odwróć kolory
mnist.data[mnist.data < 0] = 0  # Zamień wartości ujemne na 0
'''

scope = 0.2 #change to 1 once testing is done

images = mnist['data']
labels = mnist['target']
length = len(images) * scope
trainSetSize = int(length*0.8)

trainSet = []
testSet = []
trainLabels = []
testLabels = []
for i in range(int(length)):
    img = images[i]
    label = labels[i]
    generatedGraph = genGraph(img, label, 50)
    if i < trainSetSize:
        trainSet.append(generatedGraph)
    else:
        testSet.append(generatedGraph)
    
joblib.dump(trainSet, 'trainSetNoIntensity.joblib')
joblib.dump(testSet, 'testSetNoIntensity.joblib')
#joblib.dump(trainLabels, 'trainLabels.joblib')
#joblib.dump(testLabels, 'testLabels.joblib')
