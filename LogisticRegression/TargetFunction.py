from numpy.random import random


class TargetFunction(object):
    def __init__(self):
        x = 1 / 6
        self.a_2 = -x + x * random()
        self.a_1 = -x + x * random()
        self.a_0 = -x + x * random()

    def evaluate(self, x, y):
        assert (-1 <= x <= 1) and (-1 <= y <= 1)
        return 0.5 + 0.5 * (self.a_2 * x + self.a_1 * y + self.a_0 * x * y)

    def print(self):
        print("(" + str(self.a_2) + ")x^2 + (" + str(self.a_1) + ")x + (" + str(self.a_0) + ")")
