from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

from AbstractModel import Model


class RandomForest(Model):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, min_samples_leaf=2)

    def fit(self, X_train, y_train, epochs):
        print("Random Forest - fitting started", end="")
        for i in range(epochs):
            print(f"\rRandom Forest - fitting {i/epochs * 100}%", end="")
            self.model.fit(X_train, y_train)
        print("\rRandom Forest - fitting finished")


    def predict(self, X_test):
        return to_categorical(self.model.predict(X_test), self.output_size)[0]

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def summary(self):
        params = self.model.get_params()
        print("Random Forest - summary")
        print("n_estimators: " + str(params['n_estimators']))
        print("max_depth: " + str(params['max_depth']))
        print("min_samples_split: " + str(params['min_samples_split']))
        print("min_samples_leaf: " + str(params['min_samples_leaf']))
        print("random_state: " + str(params['random_state']))
