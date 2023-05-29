from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.models import Model
from keras.optimizers import Adam
from spektral.data import Dataset
from spektral.data.loaders import *
from spektral.layers import GCNConv, GlobalSumPool
from spektral.transforms import GCNFilter


class GraphModel(Model):
    batch_size = 32  # Batch size
    epochs = 1000  # Number of training epochs
    patience = 10  # Patience for early stopping
    l2_reg = 5e-4  # Regularization rate for l2
    loss_fn = SparseCategoricalCrossentropy()
    optimizer = Adam()
    trained = False
    name = ""

    def __init__(self, name, **kwargs):
        self.name = name

        super().__init__(**kwargs)
        self.conv1 = GCNConv(50, activation="elu")
        self.conv2 = GCNConv(50, activation="elu")
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(10, activation="softmax")  # MNIST has 10 classes
        self.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

    # def loadWeights(self, path):

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

    def fit(self, x_train=None, y_train=None):
        trainSet = GraphSet()
        trainSet.loadGraphs(x_train)
        trainSet.apply(GCNFilter())
        loader = PackedBatchLoader(trainSet, batch_size=10)
        super().fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=5)

    def score(self, x_test=None, y_test=None):
        testSet = GraphSet()
        testSet.loadGraphs(x_test)
        testSet.apply(GCNFilter())
        loader = BatchLoader(testSet, batch_size=10)
        loss = super().evaluate(loader.load(), steps=loader.steps_per_epoch)
        print('Test loss: {}'.format(loss))

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

    def save(self):
        # save weights to file
        return


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
