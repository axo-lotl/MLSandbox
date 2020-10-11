from SineFunction import SineFunction
from numpy.linalg import lstsq
from numpy.random import random

#fit y = ax through data, requires a parameter function f
def getLinearThroughOrigin(f):
    d1 = f.generateExample()
    d2 = f.generateExample()
    coefficients = [[d1[0]], [d2[0]]]
    results = [d1[1], d2[1]]
    slope = lstsq(coefficients, results)[0]
    return slope

def g(x):
    return 0.79 * x

def approximateBias():
    trials = 100000
    counter = 0
    f = SineFunction()
    for i in range(trials):
        x = -1 + 2 * random()
        counter += (f.evaluate(x) - g(x)) ** 2
    return counter / trials

def approximateVariance():
    trials = 100000
    counter = 0
    f = SineFunction()
    for i in range(trials):
        slope = getLinearThroughOrigin(f)
        counter += (slope - 0.79) * 0.5
    return counter / trials


print(approximateBias())