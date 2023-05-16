from sklearn.ensemble import RandomForestClassifier

from AbstractModel import Model


class RandomForest(Model):

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, min_samples_leaf=2,
                                            random_state=0)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
