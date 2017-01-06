import sys
import numpy as np

dataset_path = sys.argv[1]
weights_path = sys.argv[2]

validation_size = 200

data = np.loadtxt(dataset_path, delimiter = ',')
w = np.loadtxt(weights_path, delimiter = ',')

Y = data[:, -1]
X = data[:, :-1]
X /= np.linalg.norm(X, axis=0)

# Split the data into training/validation sets
X_train = X[:-validation_size]
X_test = X[-validation_size:]

# Split the targets into training/validation sets
Y_train = Y[:-validation_size]
Y_test = Y[-validation_size:]

with open('predictions.txt', 'w') as predictions_file:
    for i in range(validation_size):
        prediction = np.dot(X_test[i], w)
        predictions_file.write(str(prediction) + '\n')