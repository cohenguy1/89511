# Guy Cohen 304840283

import numpy as np
import matplotlib.pyplot as plt


def perceptron(x, y, validation_x, validation_y):
    max_epoch = 80
    training_size = len(x)
    validation_size = len(validation_x)

    w = np.zeros(len(x[0]))
    w_threshold = 0

    total_mistakes = 0
    for epoch in range(max_epoch):
        mistakes = 0
        validation_errors = 0

        for i in range(training_size):
            sample = x[i]
            result = np.sum(np.dot(sample, w) * y[i]) + w_threshold * y[i]
            if result <= 0:
                mistakes += 1
                w = w + y[i] * sample
                w_threshold = w_threshold + y[i]

        for i in range(validation_size):
            sample = validation_x[i]
            result = np.sum(np.dot(sample, w) * validation_y[i]) + w_threshold * validation_y[i]
            if result <= 0:
                validation_errors += 1

        print 'training: ' + str(1 - float(mistakes)/training_size) + '    validation: ' + str(1 - float(validation_errors)/validation_size)
        if mistakes == 0:
            break

        total_mistakes += mistakes

    return [w, total_mistakes]


data = np.loadtxt("training_data_1_vs_8.rs.dat.gz")
training_size = int(len(data)*0.9)
training_X = data[:training_size, 1:]
training_Y = data[:training_size, 0]

validation_X = data[training_size + 1:, 1:]
validation_Y = data[training_size + 1:, 0]

# convert label set to [-1, 1]
training_Y = (-2.0 * training_Y / 7) + 9.0 / 7
validation_Y = (-2.0 * validation_Y / 7) + 9.0 / 7

perceptron_result = perceptron(training_X, training_Y, validation_X, validation_Y)
w = perceptron_result[0]
mistakes = perceptron_result[1]

print 'perceptron mistakes: ' + str(mistakes)

tmp = 1/(1+np.exp(-10*w/w.max()))
plt.imshow(tmp.reshape(28,28),cmap="gray")
plt.draw()
plt.show()
plt.savefig("final_weight_vector")