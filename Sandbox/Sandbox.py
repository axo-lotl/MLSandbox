# void function, sorting by inputs
def merge_sort(inputs, outputs):
    if len(inputs) <= 1:
        return inputs, outputs
    elif len(inputs) == 2:
        if inputs[0] > inputs[1]:
            sorted_inputs = [inputs[1], inputs[0]]
            sorted_outputs = [outputs[1], outputs[0]]
            return sorted_inputs, sorted_outputs
    else:
        mid_index = len(inputs) // 2
        left_inputs = inputs[:mid_index]
        left_outputs = outputs[:mid_index]
        right_inputs = inputs[mid_index:]
        right_outputs = outputs[mid_index:]
        merge_sort(left_inputs, left_outputs)
        merge_sort(right_inputs, right_outputs)
        merged_data = merge(left_inputs, right_inputs, left_outputs, right_outputs)
        inputs = merged_data[0]
        outputs = merged_data[1]


# helper function for merge_sort, returns (0) merged inputs and (1) merged outputs
def merge(inputs_1, inputs_2, outputs_1, outputs_2):
    merged_inputs = []
    merged_outputs = []
    i = 0
    j = 0
    while i + j < len(inputs_1) + len(inputs_2):
        if i >= len(inputs_1):
            merged_inputs.append(inputs_2[j])
            merged_outputs.append(outputs_2[j])
            j += 1
        elif j >= len(inputs_2):
            merged_inputs.append(inputs_1[i])
            merged_outputs.append(outputs_1[i])
            i += 1
        elif inputs_1[i] < inputs_2[j]:
            merged_inputs.append(inputs_1[i])
            merged_outputs.append(outputs_1[i])
            i += 1
        else:
            merged_inputs.append(inputs_2[j])
            merged_outputs.append(outputs_2[j])
            j += 1
    return merged_inputs, merged_outputs
