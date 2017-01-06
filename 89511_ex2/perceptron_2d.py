# Guy Cohen 304840283

import numpy as np
import matplotlib.pyplot as plt


def perceptron(x, y, eta):
    max_epoch = 70
    total_num_of_mistakes = 0
    num_of_iterations = 0
    samples = len(x)

    w = np.zeros(len(x[0]) + 1)

    # part E
    for epoch in range(max_epoch):
        num_of_iterations += 1
        mistakes = 0

        # check if exists a wrong answer
        for i in range(samples):
            sample = np.array([1, x[i][0], x[i][1]])
            result = np.dot(sample, w) * y[i]
            if result <= 0:
                mistakes += 1

                # update weights
                w = w + y[i] * sample * eta

        total_num_of_mistakes += mistakes

        # can break loop since will not update anymore
        if mistakes == 0:
            break

    return [w, total_num_of_mistakes, num_of_iterations]


data = np.loadtxt('dataset.txt', delimiter = ',')
X = data[:, :-1]
Y = data[:, -1]

X_pos = []
Y_pos = []
X_neg = []
Y_neg = []

data_len = len(data)
for i in range(0, data_len):
    if data[i][-1] == 1:
        X_pos.append(data[i][0])
        Y_pos.append(data[i][1])
    else:
        X_neg.append(data[i][0])
        Y_neg.append(data[i][1])

# part D
plt.axis([-30, 30, -30, 30])
plt.plot(X_pos, Y_pos, 'bo')
plt.plot(X_neg, Y_neg, 'ro')
plt.show()

perceptron_result = perceptron(X, Y, 1.0)

W = perceptron_result[0]
perceptron_mistakes = perceptron_result[1]
perceptron_iterations = perceptron_result[2]

print 'W: ' + str(W[0]) + ' ' + str(W[1]) + ' ' + str(W[2])

x_line = np.arange(-30, 30, 0.1);
y_line = x_line * -1 * W[1] / W[2] - W[0] / W[2]
plt.plot(X_pos, Y_pos, 'bo')
plt.plot(X_neg, Y_neg, 'ro')
plt.plot(x_line, y_line)
plt.axis([-30, 30, -30, 30])
plt.show()

# write result to a file
f = open('output.txt','w')
f.write('output1: ' + str(W[0]) + ' ' + str(W[1]) + ' ' + str(W[2]) + '\n')
f.write('output2: ' + str(perceptron_mistakes) + '\n')
f.write('output3: ' + str(perceptron_iterations) + '\n')

original_data = data.copy()

# part H

np.random.shuffle(data)

perceptron_result = perceptron(X, Y, 1.0)

W = perceptron_result[0]
perceptron_mistakes = perceptron_result[1]
perceptron_iterations = perceptron_result[2]

# previous line
plt.plot(x_line, y_line)

x_line = np.arange(-30, 30, 0.1);
y_line = x_line * -1 * W[1] / W[2] - W[0] / W[2]

plt.plot(x_line, y_line)

plt.plot(X_pos, Y_pos, 'bo')
plt.plot(X_neg, Y_neg, 'ro')
plt.axis([-30, 30, -30, 30])

plt.show()
print('Shuffled data: W: ' + str(W[0]) + ' ' + str(W[1]) + ' ' + str(W[2]))

# part I
max_norm = 0
min_gamma = 100.0

# This is the perfect w (aka W star)
W = np.array([0, -0.5, 0.5])

wrong = 0
for i in range(data_len):
    norm = sum(abs(X[i]))
    if max_norm < norm:
        max_norm = norm

    sample = np.array([1.0, X[i][0], X[i][1]])
    result = np.dot(sample, W) * Y[i]

    if result < min_gamma:
        min_gamma = result

    if result < 0:
        wrong += 1

mistake_bound = (max_norm*max_norm)/(min_gamma*min_gamma)
f.write('output5: ' + str(mistake_bound) + '\n')
f.close()

X = original_data[:, :-1]
Y = original_data[:, -1]

perceptron_result = perceptron(X, Y, 0.5)

W = perceptron_result[0]
perceptron_mistakes = perceptron_result[1]
perceptron_iterations = perceptron_result[2]

print('***** Vector with learning rate *****')
print('W: ' + str(W[0]) + ' ' + str(W[1]) + ' ' + str(W[2]))
print('mistakes: ' + str(perceptron_mistakes))
print('iterations: ' + str(perceptron_iterations))