# Guy Cohen 304840283

import sys
import numpy as np

number_of_iterations = 1000
learning_rate = 0.1
regularization_parameter = 0.5

dataset_path = sys.argv[1]

data = np.loadtxt(dataset_path, delimiter = ',')

Y_train = data[:, -1]
X_train = data[:, :-1]
X_train /= np.linalg.norm(X_train, axis=0)

m = len(X_train)
dim = len(X_train[0])
w = np.zeros(dim)

for t in range(number_of_iterations):
    total_error = 0
    for i in range(m):
        total_error += (np.dot(X_train[i], w) - Y_train[i]) * (np.dot(X_train[i], w) - Y_train[i]) / m

    # print total_error

    new_w = np.zeros(dim)
    for j in range(dim):
        update_sum = 0
        for i in range(m):
            prediction = np.dot(X_train[i], w)
            diff = prediction - Y_train[i]
            update_sum += diff*X_train[i][j]
        new_w[j] = w[j] * (1 - learning_rate * 2 * regularization_parameter / m) - 2.0 * learning_rate * update_sum / m
    w = new_w

with open('weight.txt', 'w') as weights_file:
    for j in range(dim):
        weights_file.write(str(w[j]) + '\n')