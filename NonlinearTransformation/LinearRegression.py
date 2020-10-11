from TargetFunction import TargetFunction
from numpy.linalg import lstsq

def runUntransformed():
    trials = 1000
    inSampleErrorCounter = 0
    for x in range(trials):
        f = TargetFunction()
        size = 1000
        trainingSet = f.getTrainingSet(size)
        coefficients = [[0] * 3 for i in range(size)]
        results = [0] * size
        for i in range(size):
            coefficients[i][0] = 1
            coefficients[i][1] = trainingSet[i][0]
            coefficients[i][2] = trainingSet[i][1]
            results[i] = trainingSet[i][2]

        weights = lstsq(coefficients, results)[0]
        inSampleErrorCounter += f.getInSampleError(weights, trainingSet)

    print("in sample error (avg): " + str(inSampleErrorCounter / trials))

def runTransformed():
    trials = 1000
    counter = 0
    for trialnumber in range(trials):
        f = TargetFunction()
        size = 1000
        trainingSet = f.getTrainingSet(size)
        coefficients = [[0] * 6 for i in range(size)]
        results = [0] * size
        for i in range(size):
            x = trainingSet[i][0]
            y = trainingSet[i][1]
            coefficients[i][0] = 1
            coefficients[i][1] = x
            coefficients[i][2] = y
            coefficients[i][3] = x * y
            coefficients[i][4] = x ** 2
            coefficients[i][5] = y ** 2
            results[i] = trainingSet[i][2]

        weights = lstsq(coefficients, results)[0]
        counter += f.getTransformedOutSampleError(weights)

    print("out-of-sample error (avg): " + str(counter / trials))

def runTransformedOnce():
    f = TargetFunction()
    size = 1000
    trainingSet = f.getTrainingSet(size)
    coefficients = [[0] * 6 for i in range(size)]
    results = [0] * size
    for i in range(size):
        x = trainingSet[i][0]
        y = trainingSet[i][1]
        coefficients[i][0] = 1
        coefficients[i][1] = x
        coefficients[i][2] = y
        coefficients[i][3] = x * y
        coefficients[i][4] = x ** 2
        coefficients[i][5] = y ** 2
        results[i] = trainingSet[i][2]

    weights = lstsq(coefficients, results)[0]
    print(weights)

runTransformedOnce()



