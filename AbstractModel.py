class Model(object):

    def __init__(self):
        raise NotImplementedError("__init__ not implemented")

    def fit(self, X_train, y_train):
        raise NotImplementedError("fit not implemented")

    def predict(self, X_test):
        raise NotImplementedError("predict not implemented")

    def score(self, X_test, y_test):
        raise NotImplementedError("score not implemented")
