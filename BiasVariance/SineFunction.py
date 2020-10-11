from numpy.random import random
import math

class SineFunction(object):

    #domain: [-1,1]

    def __init__(self):
        pass

    def evaluate(self, x):
        assert((-1 <= x) and (x <= 1))
        return math.sin(math.pi * x)

    #returns a single data point, randomly chosen, (x, y)
    def generateExample(self):
        x = -1 + 2 * random()
        y = self.evaluate(x)
        return x, y

    #g is a METHOD that takes one variable and outputs a result; it a hypothesis
    def approximateOutOfSampleError(self, g):
        trials = 100000
        errorCounter = 0
        for i in range(trials):
            dataPoint = self.generateExample()
            x = dataPoint[0]
            errorCounter += (g(x) - dataPoint[1]) ** 2
        return errorCounter / trials







