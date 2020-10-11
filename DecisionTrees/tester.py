from TargetFunction1 import TargetFunction
from ginitree import Tree

training_set_size = 500
testing_set_size = 1000

min_size = 10
max_depth = 10

f = TargetFunction()

training_set = f.get_points(training_set_size)
testing_set = f.get_points(testing_set_size)
training_inputs = training_set[0]
training_outputs = training_set[1]
testing_inputs = testing_set[0]
testing_outputs = testing_set[1]

test_tree = Tree(training_inputs, training_outputs, min_size, max_depth)
test_tree.setup()
test_tree.print()


e_in_errors = 0
for i in range(training_set_size):
    true_result = training_outputs[i]
    guessed_result = test_tree.evaluate(training_inputs[i])
    if true_result != guessed_result:
        e_in_errors += 1

e_out_errors = 0
for i in range(testing_set_size):
    true_result = testing_outputs[i]
    guessed_result = test_tree.evaluate(testing_inputs[i])
    if true_result != guessed_result:
        e_out_errors += 1

print("E_in: " + str(e_in_errors / training_set_size))
print("E_out: " + str(e_out_errors / testing_set_size))
