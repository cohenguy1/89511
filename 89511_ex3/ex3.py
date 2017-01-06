# Guy Cohen 304840283

import math
import csv
import sys

training_path = sys.argv[1]
validation_path = sys.argv[2]
gain_measure = sys.argv[3]

epsilon = 0.0000000001

# decision tree node
class Node(object):
    def __init__(self, name, my_attributes, my_samples):
        self.name = name
        self.samples = my_samples
        self.attributes = my_attributes
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)


def init_attributes_values(my_samples):
    my_attributes_values = {}
    num_of_attributes = len(attributes)

    # map of attribute index and values for each attribute
    for attr_index in range(num_of_attributes):
        my_attributes_values[attr_index] = []

    # initialize attributes' values
    for sample in my_samples:
        for attr_index in range(num_of_attributes):
            if sample[attr_index] not in my_attributes_values[attr_index]:
                my_attributes_values[attr_index].append(sample[attr_index])

    return my_attributes_values


def get_reduced_attributes(original_attributes, attr):
    new_attributes = original_attributes[:]
    new_attributes.remove(attr)
    return new_attributes


# get samples that don't match a certain attribute
def get_reduced_samples(my_samples, attr_column, attr_value):
    my_reduced_samples = []

    for i in range(len(my_samples)):
        if my_samples[i][attr_column] == attr_value:
            my_reduced_samples.append(my_samples[i])
    return my_reduced_samples


def all_samples_with_same_tag(my_samples):
    first_tag = my_samples[0][-1]
    for i in range(len(my_samples)):
        if my_samples[i][-1] != first_tag:
            return False
    return True

# get number of samples that match a certain attribute
def get_attr_count(my_samples, column_number, value):
    attr_count = 0
    for i in range(len(my_samples)):
        if my_samples[i][column_number] == value:
            attr_count += 1.0
    return attr_count


def get_attr_prob(my_samples, column_number, value):
    attr_count = get_attr_count(my_samples, column_number, value)
    return attr_count / len(my_samples)


def get_yes_prob_with_attr(my_samples, column_number, value):
    yes_count = 0
    attr_count = 0

    for i in range(len(my_samples)):
        if my_samples[i][column_number] == value:
            attr_count += 1.0
            if my_samples[i][-1] == 'yes':
                yes_count += 1.0

    return yes_count / attr_count


def get_info_gain_entropy(my_samples, attr_num, value):
    # no samples match attribute
    if get_attr_count(my_samples, attr_num, value) == 0:
        return 0

    prob = get_yes_prob_with_attr(my_samples, attr_num, value)

    entropy = get_entropy_using_info_gain(prob)
    return entropy


def get_entropy_using_info_gain(prob):
    if prob > 1 - epsilon or prob < epsilon:
        return 0

    entropy = -prob * math.log(prob, 2) - (1 - prob) * math.log(1 - prob, 2)
    return entropy


def get_err_entropy(my_samples, attr_num, value):
    # no samples match attribute
    if get_attr_count(my_samples, attr_num, value) == 0:
        return 0

    prob = get_yes_prob_with_attr(my_samples, attr_num, value)
    entropy = get_entropy_using_err(prob)
    return entropy


def get_entropy_using_err(prob):
    entropy = min(prob, 1 - prob)
    return entropy


def get__whole_set_entropy(my_samples, measure):
    yes_prob = get_attr_prob(my_samples, -1, 'yes')

    if measure == "info-gain":
        set_entropy = get_entropy_using_info_gain(yes_prob)
    elif measure == "err":
        set_entropy = get_entropy_using_err(yes_prob)
    return set_entropy


def get_info_gain(my_samples, attr_num, attr_values, measure):
    set_entropy = get__whole_set_entropy(my_samples, measure)

    info_gain = set_entropy
    for attr_value in attr_values:

        if measure == "info-gain":
            attr_entropy = get_info_gain_entropy(my_samples, attr_num, attr_value)
        elif measure == "err":
            attr_entropy = get_err_entropy(my_samples, attr_num, attr_value)
        info_gain -= get_attr_prob(my_samples, attr_num, attr_value) * attr_entropy

    return info_gain


def get_next_tree_attr(my_samples, my_attributes, measure):
    info_gains = {}

    for attribute in my_attributes:
        attr_num = attributes.index(attribute)
        info_gains[attribute] = get_info_gain(my_samples, attr_num, attributes_values[attr_num], measure)

    max_attribute = max(info_gains, key=info_gains.get)

    return max_attribute


def append_child_nodes(my_samples, my_attributes, attribute, parent_node, nodes_to_handle):
    attribute_column = attributes.index(attribute)
    reduced_attributes = get_reduced_attributes(my_attributes, attribute)

    for attr_value in attributes_values[attribute_column]:
        reduced_samples = get_reduced_samples(my_samples, attribute_column, attr_value)
        new_attr_val_node = Node(attr_value, reduced_attributes, reduced_samples)

        parent_node.add_child(new_attr_val_node)

        # child node without samples
        if len(reduced_samples) == 0:
            add_tag_node(parent_node.samples, new_attr_val_node)
        else:
            nodes_to_handle.append(new_attr_val_node)


def build_decision_tree(my_samples, my_attributes, measure):
    max_attr = get_next_tree_attr(my_samples, my_attributes, measure)
    nodes_to_handle = []
    root = Node(max_attr, None, samples)
    append_child_nodes(my_samples, my_attributes, max_attr, root, nodes_to_handle)
    while len(nodes_to_handle) > 0:
        node = nodes_to_handle[0]

        if len(node.samples) == 0:
            # no need to handle node with no samples
            nodes_to_handle.remove(node)
        elif all_samples_with_same_tag(node.samples) or len(node.attributes) == 0:
            add_tag_node(node.samples, node)
            nodes_to_handle.remove(node)
        else:
            # get next attribute (next node in tree)
            max_attr = get_next_tree_attr(node.samples, node.attributes, measure)

            # create node
            new_attr_node = Node(max_attr, None, node.samples)
            node.add_child(new_attr_node)

            append_child_nodes(node.samples, node.attributes, max_attr, new_attr_node, nodes_to_handle)
            nodes_to_handle.remove(node)

    # reduce tree
    reduce_common_branches(root)

    return root


def reduce_common_branches(root):
    # reduce tree for branches with common decisions
    for i in range(len(attributes)):
        nodes_to_handle = [root]

        while len(nodes_to_handle) > 0:
            node = nodes_to_handle[0]

            if len(node.samples) > 0:
                if all_children_same_decision(node):
                    node.children = []
                    add_tag_node(node.samples, node)
                else:
                    for child_node in node.children:
                        nodes_to_handle.append(child_node)

            nodes_to_handle.remove(node)


def all_children_same_decision(node):
    if (node.children is None) or len(node.children) == 0:
        return False

    first_node_decision = None

    can_reduce_tree = False
    for child_node in node.children:
        if len(child_node.samples) > 0 and len(child_node.children) == 1:
            if first_node_decision is None:
                # branch decision
                first_node_decision = child_node.children[0].name

                if first_node_decision != 'yes' and first_node_decision != 'no':
                    return False

                can_reduce_tree = True
            elif child_node.children[0].name != first_node_decision:
                return False

    return can_reduce_tree


def add_tag_node(my_samples, node):
    yes_prob = get_attr_prob(my_samples, -1, 'yes')
    estimation = 'yes'
    if yes_prob < 0.5:
        estimation = 'no'
    tag_node = Node(estimation, None, node.samples)

    # add node with appropriate tag
    node.add_child(tag_node)


def print_node(tree_file, node, tabs):
    # no need to print empty node in decision tree
    if node.samples is None or len(node.samples) == 0:
        return
    for k in range(tabs - 1):
        tree_file.write('\t')
    if tabs != 0:
        tree_file.write('|-')

    tree_file.write(node.name + ' ' + ' \n')
    if (node.children is None) or len(node.children) == 0:
        return

    for child in node.children:
        print_node(tree_file, child, tabs + 1)


def print_decision_tree(root):
    with open('output_tree.txt', 'w') as tree_file:
        print_node(tree_file, root, 0)


# predict expected value by sample attributes and decision tree
def predict_sample(sample, my_decision_tree):
    current_node = my_decision_tree

    while True:
        current_attribute = current_node.name
        if current_attribute == 'yes' or current_attribute == 'no':
            return current_attribute

        if len(current_node.children) == 1:
            # only one way to go
            current_node = current_node.children[0]
            continue

        attrib_index = attributes.index(current_attribute)
        my_attrib_value = sample[attrib_index]

        # search branch with our attribute value
        for node in current_node.children:
            if node.name == my_attrib_value:
                current_node = node
                break


def predict_validation_set(validation_file_path, my_decision_tree):
    validation_set = []
    validation_count = 0.0
    validation_correct = 0

    with open("output.txt", 'w') as output_file:
        with open(validation_file_path, 'r') as validation_file:
            validation_reader = csv.reader(validation_file, delimiter='\t')
            next(validation_reader)
            for sample in validation_reader:
                validation_set.append(sample)

                expected_tag = predict_sample(sample, my_decision_tree)
                output_file.write(expected_tag + '\n')
                validation_count += 1

                if (expected_tag == sample[-1]):
                    validation_correct += 1

    # write accuracy
    with open("output_acc.txt", 'w') as accuracy_file:
        accuracy_file.write(str(validation_correct/validation_count))


# program start
tag_values = ['yes', 'no']
samples = []

with open(training_path, 'r') as train_file:
    headers = train_file.readline().rstrip('\n').split('\t')

    attributes = headers[:-1]
    tag_name = headers[-1]

    reader = csv.reader(train_file, delimiter='\t')
    for row in reader:
        samples.append(row)

# init attributes
attributes_values = init_attributes_values(samples)

decision_tree_root = build_decision_tree(samples, attributes, gain_measure)

# predict validation set
predict_validation_set(validation_path, decision_tree_root)

# print output tree with gain_measure = "err"
err_tree_root = build_decision_tree(samples, attributes, "err")

print_decision_tree(err_tree_root)