# Guy Cohen 304840283
import numpy as np
import sys

# utility function for determining hypothesis result
def get_hypo_result(sample, hypo):
    for i in range(0, dim):
        if sample[i] == 1 and hypo[i][0] == 1:
            return 0
        elif sample[i] == 0 and hypo[i][1] == 1:
            return 0

    return 1

# get input file path
file_path = sys.argv[1]

training_examples = np.loadtxt(file_path, int)
shape = training_examples.shape

dim = shape[1] - 1
num_of_samples = shape[0]

x = training_examples[:, :-1]
y = training_examples[:, dim]

# matrix of ones. first column contains negations, second contains positive literals
hypothesis = np.ones([dim, 2], int)

# iterate over the set of samples
for i in range(0, num_of_samples):
    sample = x[i]
    hypo_result = get_hypo_result(sample, hypothesis)

    # our hypothesis is no good any more
    if hypo_result == 0 and y[i] == 1:
        # remove wrong literals
        for j in range(0, dim):
            if sample[j] == 1:
                # remove negation
                hypothesis[j][0] = 0
            else:
                # remove literal
                hypothesis[j][1] = 0

first_literal = True
predicted_boolean_conj = ""

for j in range(0, dim):
    if hypothesis[j][0] == 1 or hypothesis[j][1] == 1:
        if not first_literal:
            predicted_boolean_conj += ","
        first_literal = False

        if hypothesis[j][0] == 1:
            predicted_boolean_conj += "not(x" + str(j + 1) + ")"
        else:
            predicted_boolean_conj += "x" + str(j + 1)

# write result to a file
f = open('output.txt','w')
f.write(predicted_boolean_conj)
f.close()