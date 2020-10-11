from numpy.random import permutation
from numpy.random import random
from math import exp
from math import log


# brief: transform an entire data set
# PARAMETERS:
# input_set: a 2D matrix formatted with the input set
# transform: a transform function that takes a list and returns a list; there should be no trivial variables in
# either list
# RETURNS:
# new inputs
def transform_inputs(input_set, transform):
    for inputs in input_set:
        inputs = transform(inputs)


# brief: This algorithm repeats Stochastic Gradient Descent for a fixed number of epochs.
# In each epoch, the algorithm should go through a random permutation of the data set.
# The algorithm assumes a linear model and a result normalized to a value between 0 and 1. This is typically the case
#  when you want to deal with probabilities.
# For mathematical convenience, if s is the signal, h(x) = e^s / (1 + e^s).
# The error measure in this case is given by cross-entropy error: i.e. ln(1 + e^(-ys)), where y is a real value
#  between 0 and 1 and s is the signal.
# PARAMETERS:
# input_set: the actual input_set, without any dummy weights; 2D array required of N x d, where N is the data set
#  size and d is the dimensionality
# output_set: the output set in a 1D array of size N
# eta: learning rate
# epochs: the number of epochs to use
# RETURNS:
# weights: the weight vector, with weights[0] being the threshold weight that corresponds to dummy 1
def linear_sgd(input_set_original, output_set, eta, epochs, initial_weights=None):
    input_set = input_set_original[:]
    size = len(input_set)  # first let's get a variable for the total number of data points
    for inputs in input_set:
        inputs.insert(0, 1)
    dimension = len(input_set[0])  # let's also get a convenient variable for dimensionality
    if (initial_weights is None) or (len(initial_weights) != dimension):
        weights = [-1 + 2 * random() for i in range(dimension)]
    else:
        weights = initial_weights  # many possible errors, assume user isn't dumb
    for epoch_counter in range(epochs):
        # establish a permutation
        order = [i for i in range(size)]
        order = permutation(order)
        for i in range(size): # for each data point,
            inputs = input_set[order[i]]
            output = output_set[i]
            gradient = [0] * dimension
            # now to calculate the gradient
            for j in range(dimension):
                gradient[j] = -(output * inputs[j]) / (1 + exp(output * weights[j] * inputs[j]))

            # now to update the weights
            for j in range(dimension):
                weights[j] -= eta * gradient[j]
    return weights


# gets the cross entropy error. pretty obvious
# PARAMETERS:
# weights - obvious weights
# testing_input_set - self explanatory, 2D array, no trivial values
# testing_output_set - self explanatory, 1D array
# RETURNS:
# cross-entropy error on the data set
def get_cross_entropy_error(weights, testing_input_set_original, testing_output_set):
    testing_input_set = testing_input_set_original[:]
    size = len(testing_input_set)
    for inputs in testing_input_set:
        inputs.insert(0, 1)
    assert len(weights) == len(testing_input_set[0])
    dimension = len(weights)
    sum = 0
    for i in range(size):
        inputs = testing_input_set[i]
        output = testing_output_set[i]
        inner_sum = 0  # equivalent to output * transposed weight matrix * inputs matrix
        for j in range(dimension):
            inner_sum += -output * weights[j] * inputs[j]
        sum += log(1 + exp(inner_sum))
    return sum / size;


# local algorithm: takes
def tenfold_cv(input_set, output_set, eta, epochs):
    size = len(input_set)
    data_order = [i for i in range(size)]
    data_order = permutation(data_order)
    error_counter = 0
    for i in range(10):
        cv_testing_inputs = []
        cv_testing_outputs = []
        cv_training_inputs = []
        cv_training_outputs = []
        for j in range(size):
            inputs = input_set[j]
            output = output_set[j]
            if j % 10 == i:
                cv_testing_inputs.append(inputs)
                cv_testing_outputs.append(output)
            else:
                cv_training_inputs.append(inputs)
                cv_training_outputs.append(output)
        weights = linear_sgd(cv_training_inputs, cv_training_outputs, eta, epochs)
        error = get_cross_entropy_error(weights, cv_testing_inputs, cv_testing_outputs)
        print("weights: " + str(weights) + ", error: " + str(error))
        error_counter += error

    return error_counter / 10
