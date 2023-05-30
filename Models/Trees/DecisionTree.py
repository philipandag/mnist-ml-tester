import random

from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier

from AbstractModel import Model


class DecisionTree(Model):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = DecisionTreeClassifier()
        self.best_score = 0

    def fit(self, X_train, y_train, epochs):
        for i in range(epochs):
            print(f"\rDecision Tree - fitting {i / epochs * 100:.2f}%", end="")

            max_depth = random.randint(10, 20)
            max_features = random.choice([None, 'sqrt', 'log2'])
            min_samples_split = random.randint(2, 20)
            min_samples_leaf = random.randint(1, 20)
            criterion = random.choice(['gini', 'entropy'])
            splitter = random.choice(['best', 'random'])

            model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, criterion=criterion, splitter=splitter)

            model.fit(X_train, y_train)

            score = model.score(X_train, y_train)

            if score > self.best_score:
                self.best_score = score
                self.model = model

        print(f"\rDecision Tree - fitting 100%")

    def predict(self, X_test):
        return to_categorical(self.model.predict(X_test), self.output_size)[0]

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def summary(self):
        params = self.model.get_params()
        print("-----------------------------------")
        print("Decision Tree - summary")
        print("max_depth: " + str(params['max_depth']))
        print("max_features: " + str(params['max_features']))
        print("min_samples_split: " + str(params['min_samples_split']))
        print("min_samples_leaf: " + str(params['min_samples_leaf']))
        print("criterion: " + str(params['criterion']))
        print("splitter: " + str(params['splitter']))
        print("-----------------------------------")
