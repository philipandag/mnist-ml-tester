class Confusion:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def precision(self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def accuracy(self):
        if self.tp + self.tn + self.fp + self.fn == 0:
            return 0
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def fb(self, b):
        p = self.precision()
        r = self.recall()
        if p == 0 and r == 0:
            return 0
        return (1 + b ** 2) * (p * r) / (b ** 2 * p + r)


class ConfusionMatrix:
    def __init__(self, number_of_classes):
        self.matrix = [Confusion() for _ in range(number_of_classes)]

    def add(self, predicted, actual):
        for i in range(len(self.matrix)):
            if i == predicted:
                if i == actual:
                    self.matrix[i].tp += 1
                else:
                    self.matrix[i].fp += 1
            else:
                if i == actual:
                    self.matrix[i].fn += 1
                else:
                    self.matrix[i].tn += 1

    def precision(self, i):
        return self.matrix[i].precision()

    def recall(self, i):
        return self.matrix[i].recall()

    def accuracy(self, i):
        return self.matrix[i].accuracy()

    def fb(self, b, i):
        return self.matrix[i].fb(b)
