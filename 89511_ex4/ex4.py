# Guy Cohen 304840283
import sys
import os

training_path = sys.argv[1]
test_path = sys.argv[2]

os.system("python linear_regression_train.py " + training_path)

os.system("python linear_regression_test.py " + test_path + " weight.txt true")