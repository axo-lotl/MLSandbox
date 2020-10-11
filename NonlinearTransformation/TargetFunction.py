from numpy.random import random


class TargetFunction(object):
    # domain is once again [-1,1] x [-1,1], but the target function is now known
    def __init__(self):
        pass

    def evaluate(self, x, y):
        sum = x ** 2 + y ** 2 - 0.6
        if sum >= 0:
            return 1
        else:
            return -1

    # returns a triple (x, y, result)
    def getRandomDataPoint(self):
        x = -1 + 2 * random()
        y = -1 + 2 * random()
        result = self.evaluate(x, y)
        return x, y, result

    # returns a size x 3 matrix
    def getTrainingSet(self, size):
        trainingSet = [[0] * 3 for i in range(size)]
        for i in range(size):
            dataPoint = self.getRandomDataPoint()
            for j in range(3):
                trainingSet[i][j] = dataPoint[j]

        # generate noise
        for i in range(size):
            if i % 10 == 0:
                trainingSet[i][2] *= -1  # flip result

        return trainingSet

    # evaluates the result with weights weights (w0, w1, w2), untransformed linear regression
    def evaluateWeights(self, weights, x, y):
        sum = weights[0] + (weights[1] * x) + (weights[2] * y)
        if sum >= 0:
            return 1
        else:
            return -1

    # evaluates the 6-weight transformation where the nonlinear feature vector is (1, x, y, xy, x^2, y^2),
    # w0 is still the threshold weight
    def evaluateTransformedWeights(self, weights, x, y):
        sum = weights[0] + (weights[1] * x) + (weights[2] * y) + (weights[3] * x * y) + (weights[4] * (x ** 2)) + (
        weights[5] * (y ** 2))
        if sum >= 0:
            return 1
        else:
            return -1

    # returns an approximate probability
    def getInSampleError(self, weights, trainingSet):
        errorCounter = 0
        for i in range(len(trainingSet)):
            x = trainingSet[i][0]
            y = trainingSet[i][1]
            result = trainingSet[i][2]
            hypothesizedResult = self.evaluateWeights(weights, x, y)
            if hypothesizedResult != result:
                errorCounter += 1

        return errorCounter / len(trainingSet)

    # returns an approximate probability
    def getTransformedOutSampleError(self, weights):
        trials = 1000
        errorCounter = 0
        testingSet = self.getTrainingSet(trials)
        for i in range(len(testingSet)):
            x = testingSet[i][0]
            y = testingSet[i][1]
            result = testingSet[i][2]
            hypothesizedResult = self.evaluateTransformedWeights(weights, x, y)
            if hypothesizedResult != result:
                errorCounter += 1

        return errorCounter / len(testingSet)
