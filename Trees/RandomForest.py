from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

from AbstractModel import Model


class RandomForest(Model):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, min_samples_leaf=2,
                                            random_state=0)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return to_categorical(self.model.predict(X_test), self.output_size)[0]

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
