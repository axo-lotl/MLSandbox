from math import log


# returns a dictionary of distinct values + frequencies
def get_value_frequencies(values):
    distinct_values = {}
    for value in values:
        if value not in distinct_values:
            distinct_values[value] = 1
        else:
            distinct_values[value] += 1
    return distinct_values


def get_impurity(examples):
    values = []
    for example in examples:
        values.append(example[1])
    distinct_values = get_value_frequencies(values)
    size = len(values)
    impurity = 0
    for key in distinct_values:
        proportion = distinct_values[key] / size
        impurity += proportion * (1 - proportion)
    return impurity


def get_information_gain(examples):
    values = []
    for example in examples:
        values.append(example[1])
    distinct_values = get_value_frequencies(values)
    size = len(values)
    information_gain = 0
    for key in distinct_values:
        proportion = distinct_values[key] / size
        information_gain += -proportion * log(proportion, 2)
    return information_gain


# each example is an ordered pair (input, output); both input and output are real-valued; sort by inputs
def merge_sort(examples):
    if len(examples) <= 1:
        return examples
    elif len(examples) == 2:
        if examples[0][0] > examples[1][0]:
            new_examples = [examples[1], examples[0]]
            return new_examples
        else:
            return examples
    else:
        boundary = len(examples) // 2
        left_examples = examples[:boundary]
        right_examples = examples[boundary:]
        sorted_left_examples = merge_sort(left_examples)
        sorted_right_examples = merge_sort(right_examples)
        return merge(sorted_left_examples, sorted_right_examples)


# merges two (pre-sorted by inputs) example sets into a (sorted-by-input) example set
def merge(examples_1, examples_2):
    i = 0
    j = 0
    examples = []
    while i + j < len(examples_1) + len(examples_2):
        if i >= len(examples_1):  # if we have used everything in examples_1, take from examples_2
            examples.append(examples_2[j])
            j += 1
        elif j >= len(examples_2):  # if we have used everything in examples_2, take from examples_1
            examples.append(examples_1[i])
            i += 1
        elif examples_1[i][0] < examples_2[j][0]:  # compare to find the smaller INPUT
            examples.append(examples_1[i])
            i += 1
        else:
            examples.append(examples_2[j])
            j += 1
    return examples


# calculates the best way (highest Gini impurity loss) to split the list into two NONEMPTY smaller lists
# the lists are split based on whether the input falls below or above a certain "threshold"
# returns:
# (0) that threshold; will return None if there is no way to get an impurity loss of more than 0
# (1) the Gini impurity loss; will return 0 if there is no way to get an impurity loss of more than 0
def get_best_split(examples):
    sorted_examples = merge_sort(examples)  # first we have to sort our examples by input
    best_i = None
    best_impurity_loss = 0
    current_impurity = get_impurity(examples)
    for i in range(1, len(sorted_examples)):  # disregard trivial cases of making 0-length lists
        # first check that we can even set a threshold
        if examples[i - 1][0] == examples[i][0]:
            pass  # if this happens there is no valid threshold, even if there is good impurity loss
        else:
            left_impurity = get_impurity(sorted_examples[:i])
            right_impurity = get_impurity(sorted_examples[i:])
            impurity_loss = current_impurity - ((left_impurity + right_impurity) / 2)
            if impurity_loss > best_impurity_loss:
                best_impurity_loss = impurity_loss
                best_i = i

    if best_i is not None:  # some split is possible
        threshold = (examples[best_i - 1][0] + examples[best_i][0]) / 2
        return threshold, best_impurity_loss
    else:
        return None, best_impurity_loss


