# Guy Cohen 304840283

import numpy as np
import matplotlib.pyplot as plt


def perceptron(x, y):
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
                w = w + y[i] * sample

        total_num_of_mistakes += mistakes

        # can break loop since will not update anymore
        if mistakes == 0:
            break

    return [w, total_num_of_mistakes, num_of_iterations]


def find_w_star(x, y):
    total_num_of_mistakes = 0
    samples = len(x)

    w_star = np.zeros(len(x[0]) + 1)

    mistakes = 1

    while mistakes > 0:
        np.random.shuffle(X)
        mistakes = 0

        # check if exists a wrong answer
        for i in range(samples):
            sample = np.array([1, x[i][0], x[i][1]])
            result = np.dot(sample, w_star) * y[i]
            if result <= 0:
                mistakes += 1

                # update weights
                w_star = w_star + y[i] * sample
                w_star = w_star / sum(abs(w_star))
                sum1 = sum(abs(w_star))

        total_num_of_mistakes += mistakes

        # can break loop since will not update anymore
        if mistakes == 0:
            break

    return w_star

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

perceptron_result = perceptron(X, Y)

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
f.close()

# part H

np.random.shuffle(data)

perceptron_result = perceptron(X, Y)

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
print('W: ' + str(W[0]) + ' ' + str(W[1]) + ' ' + str(W[2]))

# part I
max_distance = 0

for i in range(data_len):
    norm = sum(abs(X[i]))
    if max_distance < norm:
        max_distance = norm

w_star = find_w_star(X, Y)
print('w_star: ' + str(w_star[0]) + ' ' + str(w_star[1]) + ' ' + str(w_star[2]))

print norm