import scipy.io
import numpy as np
import time
import matplotlib.pyplot as plt

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

# initializing all 10 thetas with 5
theta = [0.05]*10

# creating an array for storing elapsed time and function value at that point
func_value = list()
elapsed_time = list()

XMean = np.mean(X)
XStd = np.std(X)
XNormalised = np.divide(X - XMean, XStd)

# setting descent parameter as 0.1
n = 0.001

# running our gradient descent algorithm 10 times and
# recording the function value and elapsed time at each of these times
temp_func_value = 0
for k in range(0, 10):
    for i in range(0, 10000):
        for d in range(0, 784):
            temp_func_value += func(XNormalised[i, d], YNew[i, k], theta[k])

print("start value of function  "+ str(temp_func_value))
func_value.append(temp_func_value)


run_frequency = 20
for freq in range(0, run_frequency):
    # list of function values for 10 values of k (o to 9)
    func_value_per_iteration = list()
    for k in range(0, 10):
        temp_func_value = 0
        summation_term = 0
        # finding the function gradient to find new value of theta[k]
        for i in range(0, 10000):
            for d in range(0, 784):
                summation_term += funcdash(XNormalised[i, d], YNew[i, k], theta[k])
        theta[k] = theta[k] - (n/10000)*summation_term

        # Calculating the function value with new value of theta[k]
        for i in range(0, 10000):
            for d in range(0, 784):
                temp_func_value += func(XNormalised[i, d], YNew[i, k], theta[k])

        func_value_per_iteration.append(temp_func_value)
        print("iteration: " + str(k) + " completed")

    func_value.append(np.sum(func_value_per_iteration))
    elapsed_time.append(time.time() - start_time)


print("test")

# plotting a graph between elapsed time and function value
plot1, = plt.plot(elapsed_time, func_value, 'ro', label='Decreasing Function Value (batch gradient descent)')
plt.axis([900, 50000, 4000000, 3500000])
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Function Value')
plt.legend(handles=[plot1])
plt.show()