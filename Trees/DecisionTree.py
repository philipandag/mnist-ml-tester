from sklearn.tree import DecisionTreeClassifier

from AbstractModel import Model


class DecisionTree(Model):

    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5, random_state=0)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
