import sys
import numpy as np

print_to_stdout = False
if len(sys.argv) > 3:
    print_to_stdout = True

test_path = sys.argv[1]
weights_path = sys.argv[2]

data = np.loadtxt(test_path, delimiter = ',')
w = np.loadtxt(weights_path, delimiter = ',')

Y_test = data[:, -1]
X_test = data[:, :-1]
X_test /= np.linalg.norm(X_test, axis=0)

m = len(X_test)

if print_to_stdout:
    for i in range(m):
        prediction = np.dot(X_test[i], w)
        print(str(prediction))

else:
    predictions_file = open('predictions.txt', 'w')
    try:
        for i in range(m):
            prediction = np.dot(X_test[i], w)
            predictions_file.write(str(prediction) + '\n')
    finally:
        predictions_file.close()
