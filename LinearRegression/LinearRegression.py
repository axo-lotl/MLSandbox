from TargetFunction import TargetFunction
from numpy.linalg import lstsq

trials = 1000
eInCounter = 0
eOutCounter = 0

for i in range(trials):
    f = TargetFunction()

    # debugging only
    # print("y = " + str(f.m) + "x + " + str(f.b))

    trainingSet = f.getTrainingSet(100)
    coefficients = trainingSet[0]
    results = trainingSet[1]

    weights = lstsq(coefficients, results)[0]
    eInCounter += f.getEIn(weights, coefficients, results)
    eOutCounter += f.getEOut(weights)

print("average E_in: " + str(eInCounter / trials))
print("average E_out: " + str(eOutCounter / trials))