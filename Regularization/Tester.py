from numpy import matrix
from numpy import identity
from numpy.linalg import lstsq
from numpy.linalg import inv


def get_data_from_file(filename):
    file = open(filename, 'r')
    data_set = []
    for line in file:
        words = line.split()
        floats = [0] * len(words)
        for i in range(len(words)):
            floats[i] = float(words[i])
        data_set.append(floats)
    return data_set


# returns input matrix, output matrix
def get_formatted_data(training_set):
    transformed_inputs = [[0] * 8 for i in range(len(training_set))]
    outputs = [[0] * 1 for i in range(len(training_set))]
    for i in range(len(training_set)):
        outputs[i][0] = training_set[i][2]
        transformed_inputs[i][0] = 1
        x1 = training_set[i][0]
        x2 = training_set[i][1]
        transformed_inputs[i][1] = x1
        transformed_inputs[i][2] = x2
        transformed_inputs[i][3] = x1 ** 2
        transformed_inputs[i][4] = x2 ** 2
        transformed_inputs[i][5] = x1 * x2
        transformed_inputs[i][6] = abs(x1 - x2)
        transformed_inputs[i][7] = abs(x1 + x2)
    return transformed_inputs, outputs


def get_error(weights, inputs_matrix, outputs_matrix):
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

training_set = get_data_from_file('in.dta.txt')
testing_set = get_data_from_file('out.dta.txt')
training_set_formatted = get_formatted_data(training_set)
testing_set_formatted = get_formatted_data(testing_set)

Z = matrix(training_set_formatted[0])
y = matrix(training_set_formatted[1])
I = matrix(identity(8))

for k in range(-2,3):
    reg_factor = 10 ** k
    print("lambda = " + str(reg_factor))
    t_1 = Z.transpose() * Z
    t_2 = reg_factor * I
    t_3 = inv(t_1 + t_2)
    w_reg = t_3 * Z.transpose() * y
    weights = w_reg.tolist()
    print('in sample error: ' + str(get_error(weights, training_set_formatted[0], training_set_formatted[1])))

    print('out of sample error: ' + str(get_error(weights, testing_set_formatted[0], testing_set_formatted[1])))





