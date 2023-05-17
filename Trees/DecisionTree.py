from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier

from AbstractModel import Model


class DecisionTree(Model):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5, random_state=0)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return to_categorical(self.model.predict(X_test), self.output_size)[0]

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
