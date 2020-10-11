from numpy.random import random


# takes 5 reals between 0 and 1 and returns a boolean expression

class TargetFunction(object):
    def __init__(self):
        pass

    def evaluate(self, input_vector):
        assert len(input_vector) >= 5
        score = 0
        for i in range(5):
            if input_vector[i] < 0.1 * (i + 1):
                score += 5 - i
        return score > 7

    def get_points(self, size):
        inputs = []
        outputs = []
        for i in range(size):
            input_vector = [random() for j in range(5)]
            output = self.evaluate(input_vector)
            inputs.append(input_vector)
            outputs.append(output)
        return inputs, outputs
