import random

from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

from AbstractModel import Model


class RandomForest(Model):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = RandomForestClassifier()
        self.best_score = 0

    def fit(self, X_train, y_train, epochs):
        for i in range(epochs):
            print(f"\rRandom Forest - fitting {i / epochs * 100}%", end="")

            n_estimators = random.randint(1, 20)
            max_depth = random.randint(1, 20)
            min_samples_split = random.randint(2, 20)
            min_samples_leaf = random.randint(1, 20)
            criterion = random.choice(['gini', 'entropy'])

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, criterion=criterion)

            model.fit(X_train, y_train)

            score = model.score(X_train, y_train)

            if score > self.best_score:
                self.best_score = score
                self.model = model

        print(f"\rRandom Forest - fitting 100%")

    def predict(self, X_test):
        return to_categorical(self.model.predict(X_test), self.output_size)[0]

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def summary(self):
        params = self.model.get_params()
        print("-----------------------------------")
        print("Random Forest - summary")
        print("n_estimators: " + str(params['n_estimators']))
        print("max_depth: " + str(params['max_depth']))
        print("min_samples_split: " + str(params['min_samples_split']))
        print("min_samples_leaf: " + str(params['min_samples_leaf']))
        print("criterion: " + str(params['criterion']))
        print("-----------------------------------")
