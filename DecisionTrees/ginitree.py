import algorithms


class Node(object):
    # GENERAL PROPERTIES
    # inputs: inputs of the data set the Node has
    # outputs: outputs of the data set the Node has
    # tier: the level/tier of the Node in the tree; 0 for root, children have tier 1 more than their parents
    # min_size: if the Node has less examples than this, it does not split and converts to a leaf
    # max_depth: if the Node's tier is equal to this, it does not split and converts to a leaf
    #
    # BRANCH PROPERTIES (all None if the Node is not a branch)
    # left: left child Node
    # right: right child Node
    # attribute: index of a specific attribute in an input vector
    # threshold: if attribute < threshold, go to the left child; otherwise, go to the right child
    #
    # LEAF PROPERTIES (all None if the Node is not a leaf)
    # value: literally an output value

    def __init__(self, inputs, outputs, tier, min_size, max_depth):
        self.inputs = inputs
        self.outputs = outputs
        self.tier = tier
        self.min_size = min_size
        self.max_depth = max_depth
        self.is_leaf = False

        self.left = None
        self.right = None
        self.attribute = None
        self.threshold = None

        self.value = None

    def chain(self):  # can set up the entire Tree from the root node
        self.setup()
        if self.left is not None:
            self.left.chain()
        if self.right is not None:
            self.right.chain()

    # returns a string description
    def get_description(self):
        if self.value is None:  # i.e. if the Node is a branch
            return "BRANCH: left condition = attribute #" + str(self.attribute) + " < " + str(self.threshold)
        else:  # Node is a leaf
            return "LEAF: value = " + str(self.value)

    def print_tiered_description(self):
        print("-" * self.tier + self.get_description())

    # prints all children as well
    def print_all(self):
        self.print_tiered_description()
        if self.left is not None:
            self.left.print_all()
        if self.right is not None:
            self.right.print_all()

    def become_leaf(self):
        assert len(self.outputs) > 0
        # just get the most frequent output
        output_frequencies = algorithms.get_value_frequencies(self.outputs)
        most_frequent_output = None
        occurrences = 0
        for output in output_frequencies:
            if output_frequencies[output] > occurrences:
                most_frequent_output = output
                occurrences = output_frequencies[output]
        self.value = most_frequent_output
        self.left = None
        self.right = None
        self.attribute = None
        self.threshold = None

    def setup(self):
        # check for trivial leafing requirements
        if self.tier >= self.max_depth:
            self.become_leaf()
        elif len(self.outputs) < self.min_size:
            self.become_leaf()

        # attempt to find the best attribute/threshold combo to maximize Gini impurity loss
        assert len(self.inputs) > 0 and len(self.inputs) == len(self.outputs)
        size = len(self.inputs)
        dimensions = len(self.inputs[0])

        # these variables track progress
        best_attribute = None
        best_threshold = None
        best_impurity_loss = 0

        for d in range(dimensions):
            # format the examples
            examples = []
            for i in range(size):
                examples.append([self.inputs[i][d], self.outputs[i]])
            best_split = algorithms.get_best_split(examples)
            threshold = best_split[0]
            impurity_loss = best_split[1]
            if impurity_loss > best_impurity_loss:
                best_attribute = d
                best_threshold = threshold
                best_impurity_loss = impurity_loss

        if best_attribute is None:  # couldn't find anything? go leaf yourself
            self.become_leaf()
        else:
            self.value = None
            self.attribute = best_attribute
            self.threshold = best_threshold
            # create leaf nodes; separate data into "left" and "right" nodes
            left_inputs = []
            left_outputs = []
            right_inputs = []
            right_outputs = []
            for i in range(size):  # for every example...
                if self.inputs[i][self.attribute] < self.threshold:  # go to the left if this condition si method
                    left_inputs.append(self.inputs[i])
                    left_outputs.append(self.outputs[i])
                else:  # go to the right otherwise
                    right_inputs.append(self.inputs[i])
                    right_outputs.append(self.outputs[i])
            # left_inputs, left_outputs, right_inputs, right_outputs should be filled
            self.left = Node(left_inputs, left_outputs, self.tier + 1, self.min_size, self.max_depth)
            self.right = Node(right_inputs, right_outputs, self.tier + 1, self.min_size, self.max_depth)


class Tree(object):
    # MEMBER VARIABLES:
    # root: the root node of the tree

    def __init__(self, inputs, outputs, min_size, max_depth):
        self.root = Node(inputs, outputs, 0, min_size, max_depth)

    def setup(self):
        self.root.chain()

    # evaluates the result of an input_vector
    def evaluate(self, input_vector):
        current = self.root  # variable that holds the current node
        while current.value is None:  # while "current" does not have a value, i.e. "current" is a branch
            if input_vector[current.attribute] < current.threshold:
                current = current.left
            else:
                current = current.right
        # finally...
        return current.value

    def print(self):
        self.root.print_all()
