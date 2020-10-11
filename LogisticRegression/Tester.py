from TargetFunction import TargetFunction
from numpy.random import random
import Algorithms

f = TargetFunction()
batch_size = 1000
validation_size = 1000
eta = 0.001
epochs = 250

input_set = []
output_set = []
testing_input_set = []
testing_output_set = []


def transform(inputs):
    x = inputs[0]
    y = inputs[1]
    return [x, y, x * y, x ** 2, y ** 2, (x ** 2) * y, x * (y ** 2), x ** 3, y ** 3]


for i in range(batch_size):
    preliminary = [-1 + 2 * random(), -1 + 2 * random()]
    inputs = transform(preliminary)
    input_set.append(inputs)
    output_set.append(f.evaluate(preliminary[0], preliminary[1]))

for i in range(validation_size):
    preliminary = [-1 + 2 * random(), -1 + 2 * random()]
    inputs = transform(preliminary)
    testing_input_set.append(inputs)
    testing_output_set.append(f.evaluate(preliminary[0], preliminary[1]))

f.print();
weights = Algorithms.linear_sgd(input_set, output_set, eta, epochs)
print(weights)
print(str(Algorithms.get_cross_entropy_error(weights, testing_input_set, testing_output_set)))
