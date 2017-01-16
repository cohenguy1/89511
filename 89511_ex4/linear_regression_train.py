# Guy Cohen 304840283

import sys
import numpy as np

number_of_iterations = 1000
learning_rate = 0.1
regularization_parameter = 0.5

dataset_path = sys.argv[1]

validation_size = 200

data = np.loadtxt(dataset_path, delimiter = ',')

Y = data[:, -1]
X = data[:, :-1]
X /= np.linalg.norm(X, axis=0)

# Split the data into training/validation sets
X_train_raw = X[:-validation_size]
X_test_raw = X[-validation_size:]

# Split the targets into training/validation sets
Y_train = Y[:-validation_size]
Y_test = Y[-validation_size:]

X_train = []
for sample in X_train_raw:
    X_train.append(np.append(1, sample))

m = len(X_train)
dim = len(X_train[0])
w = np.zeros(dim)

for epoch in range(number_of_iterations):
    update_sum = np.zeros(dim)
    for i in range(m):
        update_sum += np.dot(np.dot(X_train[i], w) - Y_train[i], X_train[i])
    new_w = w - 2.0 * learning_rate * update_sum / m - w * learning_rate * 2.0 * regularization_parameter
    w = new_w

weights_file = open('weight.txt', 'w')
try:
    for j in range(dim):
        weights_file.write(str(w[j]) + '\n')
        print (str(w[j]))
finally:
    weights_file.close()