
from keras.models import Model
from keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool
from spektral.data import BatchLoader
from spektral.data.loaders import *
from spektral.data import Dataset
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Adam
from spektral.transforms import GCNFilter
import tensorflow as tf
from math import sqrt
import numpy as np
import spektral.data as sd



class GraphModel(Model):
    batch_size = 32  # Batch size
    epochs = 1000  # Number of training epochs
    patience = 10  # Patience for early stopping
    l2_reg = 5e-4  # Regularization rate for l2
    loss_fn = SparseCategoricalCrossentropy()
    optimizer = Adam()
    def __init__(self, **kwargs):   
        super().__init__(**kwargs)
        self.conv1 = GCNConv(50, activation="elu")
        self.conv2 = GCNConv(50, activation="elu")
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(10, activation="softmax")  # MNIST has 10 classes
        self.compile(optimizer = self.optimizer, loss = self.loss_fn, metrics=['accuracy'])
        

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output
    
    def fit(self, x_train):
        trainSet = GraphSet()
        trainSet.loadGraphs(x_train)
        trainSet.apply(GCNFilter())
        loader = PackedBatchLoader(trainSet, batch_size=10)
        super().fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=25)
        
    def fit(self, x_train, y_train):
        graphs = []
        for i in range(len(x_train)):
            img = x_train[i]
            label = y_train[i]
            generatedGraph = genGraph(img, label, 50)
            graphs.append(generatedGraph)
    
    def score(self,x_test):
        testSet = GraphSet()
        testSet.loadGraphs(x_test)
        testSet.apply(GCNFilter())
        loader = BatchLoader(testSet, batch_size=10)
        loss = super().evaluate(loader.load(), steps=loader.steps_per_epoch)
        print('Test loss: {}'.format(loss))
        
    def fit(self, x_test, y_test):
        graphs = []
        for i in range(len(x_test)):
            img = x_test[i]
            label = y_test[i]
            generatedGraph = genGraph(img, label, 50)
            graphs.append(generatedGraph)
    
    def predict(self, example):
        dataset = GraphSet()
        dataset.loadGraphs(example)
        dataset.apply(GCNFilter())
        loader = BatchLoader(dataset)
        return super().predict(loader.load(), steps=loader.steps_per_epoch)

    @tf.function
    def train_on_batch(self, inputs, target):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss_fn(target, predictions) + sum(self.losses)
            acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, acc

class GraphSet(Dataset):
    graphs = []
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def read(self):
        return self.graphs
    
    def loadGraphs(self, g):
        if type(g) is list:
            self.graphs = g
        else:
            self.graphs = [g]

    def download(self):
        return


def genGraph(img, label, size=25):
    img = np.asarray(img.astype(np.float32))
    tempImg = []
    for h in range(int(sqrt(img.size))):
        row = []
        for w in range(int(sqrt(img.size))):
            row.append(img[h*8+w])
        tempImg.append(row)
    img = np.asarray(tempImg)


    import scipy.ndimage
    from skimage.segmentation import slic
    from scipy.spatial.distance import cdist


    superpixels = slic(img,n_segments=size, compactness=0.25, channel_axis=None)
    #print(superpixels)
    sp_indices = np.unique(superpixels)
    n_sp = len(sp_indices)  

    sp_intensity = np.zeros((n_sp, 1), np.float32)
    sp_coord = np.zeros((n_sp, 2), np.float32)  # row, col
    for seg in sp_indices:
        mask = superpixels == seg
        sp_intensity[seg-1] = np.mean(img[mask])
        sp_coord[seg-1] = np.array(scipy.ndimage.measurements.center_of_mass(mask))


    #t(sp_intensity)


    sp = superpixels
    xlen = len(superpixels)
    ylen = len(superpixels[0])
    edges = []
    def tryAddEdge(xind, yind):
        if xind<xlen and xind >= 0 and yind<ylen and yind >= 0 and sp[xind][yind] != ind:
            if(ind < sp[xind][yind]):
                edges.append( (ind, sp[xind][yind]) )
            else:
                edges.append( (sp[xind][yind], ind) )

    for ind in sp_indices:
        for x in range(xlen):
            for y in range(ylen):
                if sp[x][y] == ind:
                    tryAddEdge(x+1,y)
                    tryAddEdge(x-1,y)
                    tryAddEdge(x,y+1)
                    tryAddEdge(x, y-1)
                    tryAddEdge(x+1,y+1)
                    tryAddEdge(x+1, y-1)
                    tryAddEdge(x-1, y+1)
                    tryAddEdge(x-1, y-1)

    '''
    #visualise sliced
    vis = []
    for x in range(len(sp)):
        row = []
        for y in range(len(sp[0])):
            pixId = sp[x][y]
            row.append(sp_intensity[pixId-1])
        vis.append(row)
    plt.figure()
    plt.imshow(vis)
    '''
        


    edges = list(set(edges))
    edges = np.asarray(edges)
    edges = edges-1
    #print(edges)

    edgeWeights = []
    for E in edges:
        p1x = sp_coord[E[0]][0]
        p2x = sp_coord[E[1]][0]
        p1y = sp_coord[E[0]][1]
        p2y = sp_coord[E[1]][1]
        dist = sqrt(pow(p1x - p2x, 2) + pow(p1y - p2y, 2))
        edgeWeights.append(dist)

    '''
    G = nx.Graph()

    for x in sp_intensity:
        G.add_node(x[0])
        
    for E in range(len(edges)):
        N1 = sp_intensity[(edges[E])[0]][0]
        N2 = sp_intensity[(edges[E])[1]][0]
        if N1 != N2:
            G.add_edge(N1, N2, weight=edgeWeights[E])

    plt.figure()for E in edges:
        p1 = sp_coord[E[0]]
        p2 = sp_coord[E[1]]
        xdist2 = pow((p1[0] - p2[0]),2)
        ydist2 = pow((p1[1] - p2[1]),2)
        dist = sqrt(xdist2+ydist2)
    G = G.to_undirected()
    nx.draw_networkx(G)
    plt.show()
    '''

    #adjacency matrix for graph
    A = np.zeros((n_sp, n_sp))
    for E in range(len(edges)):
        x = (edges[E])[0]
        y = (edges[E])[1]
        #A[x][y] = edgeWeights[E]
        #A[y][x] = edgeWeights[E]
        A[x][y] = 1
        A[y][x] = 1
    
    from scipy import sparse
    A = sparse.csr_matrix(A)
    #X = sp_intensity
    X = np.ones(sp_intensity.shape)
    E = None
    Y = label
    #Y = None
    
    g = sd.Graph(X,A,E,Y)
    return g