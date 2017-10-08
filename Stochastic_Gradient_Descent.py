import scipy.io
import numpy as np
import time
import matplotlib.pyplot as plt
from random import *

start_time = time.time()

mat_contents = scipy.io.loadmat('hw1data.mat')
X = mat_contents['X'].astype(np.float64)
Y = mat_contents['Y'].astype(np.float64)

# converting Y to a "one-hot encoded" vector
YNew = np.zeros((10000, 10))
for i in range(0, 10000):
    digit = int(Y[i])
    YNew[i, digit] = 1


def func(x, y, theta_val):
    return (1/2)*(y)*np.power((x-theta_val), 2)


def funcdash(x, y, theta_val):
    return -y*(x-theta_val)


# dividing the training data into 10 buckets
def get_x_given_y(X, Y, y):
    return [x for index, x in enumerate(X) if Y[index] == y]

# initializing all 10 thetas with 5
theta = [50]*10

# setting descent parameter as 0.1
n = 0.000000001

# creating an array for storing elapsed time and function value at that point
func_value = list()
elapsed_time = list()

XMean = np.mean(X)
XStd = np.std(X)
XNormalised = np.divide(X - XMean, XStd)

# running our gradient descent algorithm a certain number times and
# recording the function value and elapsed time at each of these times

run_frequency = 250
for freq in range(0, run_frequency):
    # list of function values for 10 values of k (o to 9)
    func_value_per_iteration = list()

    #Updating the theta values using a random sample from kth bucket of training data
    for k in range(0, 10):
        temp_func_value = 0
        summation_term = 0
        # put the code of selecting one random training data from the kth bucket
        XBucketed = get_x_given_y(XNormalised, Y, k)
        size = XBucketed.__len__()
        randomIndex = randrange(0, size)
        x_random_choice = XBucketed[randomIndex]
        y_random_choice = np.zeros(10)
        y_random_choice[k] = 1
        for d in range(0, 784):
            summation_term += funcdash(x_random_choice[d], y_random_choice[k], theta[k])
        theta[k] = theta[k] - (n)*summation_term #we multiply (n/size) by size to scale the gradient by

        # Calculating the function value with new value of theta[k]
        for i in range(0, 10000):
            for d in range(0, 784):
                temp_func_value += func(XNormalised[i, d], YNew[i, k], theta[k])

        func_value_per_iteration.append(temp_func_value)

    func_value.append(np.sum(func_value_per_iteration))
    elapsed_time.append(time.time() - start_time)

# plotting a graph between elapsed time and function value
plot1, = plt.plot(elapsed_time, func_value, 'ro', label='Function Value')
plt.axis([0, 15, 9600000, 9900000])
plt.xlabel('Elapsed Time')
plt.ylabel('Function Value')
plt.legend(handles=[plot1])
plt.show()