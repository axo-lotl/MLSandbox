from numpy import matrix
from numpy import identity
from numpy.linalg import lstsq
from numpy.linalg import inv


# returns 2D arrays of inputs and outputs
def get_data_from_file(filename):
    file = open(filename, 'r')
    inputs = []
    outputs = []
    for line in file:
        words = line.split()
        floats = [0] * len(words)
        for i in range(len(words)):
            floats[i] = float(words[i])
        inputs.append(floats[0:2])
        outputs.append(floats[2:3])
    return inputs, outputs


# transform is a function that returns transformed inputs
# returns a new input_set
# adds the dummy 1s as well
def get_transformed_inputs(input_set, transform_function):
    transformed_input_set = []
    for i in range(len(input_set)):
        inputs = input_set[i]
        dummy = [1]
        transformed_inputs = transform_function(inputs)
        transformed_input_set.append(dummy + transformed_inputs)
    return transformed_input_set


# returns weights as a list
# in normal circumstances w0 is the threshold, multiplied by the dummy 1 in all inputs
def linear_regression(input_set, output_set):
    return lstsq(input_set, output_set)[0]


def check_error(weights, inputs_matrix_u, outputs_matrix, transform_function):
    inputs_matrix = get_transformed_inputs(inputs_matrix_u, transform_function)
    set_size = len(outputs_matrix)
    error_counter = 0
    for i in range(set_size):
        sum = 0
        for j in range(len(weights)):
            sum += weights[j][0] * inputs_matrix[i][j]
        predicted_result = 0
        if sum > 0:
            predicted_result = 1
        elif sum < 0:
            predicted_result = -1
        if predicted_result != outputs_matrix[i][0]:
            error_counter += 1
    return error_counter / set_size


def indexed_transform(inputs, index):
    x1 = inputs[0]
    x2 = inputs[1]
    t_list = [1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)]
    r_list = t_list[0:index + 1]
    return r_list


full_training_data = get_data_from_file('in.dta.txt')
full_testing_data = get_data_from_file('out.dta.txt')
training_inputs = full_training_data[0]
training_outputs = full_training_data[1]
testing_inputs = full_testing_data[0]
testing_outputs = full_testing_data[1]

for i in range(3, 8, 1):
    def transform(inputs):
        return indexed_transform(inputs, i)


    print("k = " + str(i))
    actual_training_inputs = training_inputs[0:25]
    actual_training_outputs = training_outputs[0:25]
    validation_inputs = training_inputs[25:35]
    validation_outputs = training_outputs[25:35]

    weights = linear_regression(get_transformed_inputs(actual_training_inputs, transform), actual_training_outputs)
    print('validation error: ' + str(check_error(weights, validation_inputs, validation_outputs, transform)))
    print('out-of-sample error: ' + str(check_error(weights, testing_inputs, testing_outputs, transform)))
