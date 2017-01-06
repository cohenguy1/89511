# Guy Cohen 304840283

import sys
import numpy as np

number_of_iterations = 1000
learning_rate = 0.1
regularization_parameter = 0.5

validation_size = 200

dataset_path = sys.argv[1]

data = np.loadtxt(dataset_path, delimiter = ',')

Y = data[:, -1]
X = data[:, :-1]
X /= np.linalg.norm(X, axis=0)

# Split the data into training/validation sets
X_train = X[:-200]
X_test = X[-200:]

# Split the targets into training/validation sets
Y_train = Y[:-200]
Y_test = Y[-200:]

