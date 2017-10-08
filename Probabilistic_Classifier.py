import scipy.io as sio
import numpy as np
import numpy.linalg as nlin
import math

def get_x_given_y(X, Y, y):
    return [x for index, x in enumerate(X) if Y[index] == y]

def get_mean_std_deviation(X, Y): 
    mean = [None for i in range(10)]
    std = [None for i in range(10)]
    
    for i in range(10):
        x_i = get_x_given_y(X, Y, i)
        mean[i] = np.mean(x_i, axis=0)
        std[i] = np.std(x_i, axis=0)

    return mean, std

def pY(Y, y):
    return (Y == y).sum()/Y.shape[0]

def get_transformed_data_vector(x, yVal, droppedIndices):
    feature_size = x.shape[0]
    newX = []

    for i in range(feature_size):
        if i not in droppedIndices[yVal]:
            newX.append(x[i])

    return np.array(newX)

def get_transformed_data_matrix(X, Y, droppedIndices):
    data_size = X.shape[0]
    feature_size = X.shape[1]
    newX_train = []

    for i in range(data_size):
        newX_train.append(get_transformed_data_vector(X[i], Y[i], droppedIndices))

    return newX_train

if __name__ == "__main__":
    data = sio.loadmat("hw1data.mat")

    X = data["X"]
    Y = data["Y"]
    train_2_test_proportion = 70/100
    data_size = int(train_2_test_proportion * X.shape[0])
    feature_size = X.shape[1]
    X_train = np.array(X[0:data_size, :])
    Y_train = np.array(Y[0:data_size, 0])
    testX = np.array(X[data_size:, :])
    testY = np.array(Y[data_size:, 0])
    testSize = testY.shape[0]

    mean, std = get_mean_std_deviation(X_train, Y_train)

    dropList = [[] for i in range(10)]

    # Find all the features to drop which have 0 standard deviation
    for i in range(10):
        for j in range(feature_size):
            if std[i][j] == 0:
                dropList[i].append(j)

    # Get the new training data after truncating the features
    newX_train = get_transformed_data_matrix(X_train, Y_train, dropList)

    mean, std = get_mean_std_deviation(newX_train, Y_train)

    # Normalize the training data
    for i in range(data_size):
        newX_train[i] = np.divide(newX_train[i] - mean[Y_train[i]], std[Y_train[i]])

    # Get new standard deviation, Mean should be zero
    mean, std = get_mean_std_deviation(newX_train, Y_train)

    cov = [None for i in range(10)]
    
    # Get the covariance matrix using numpy's cov function
    for i in range(10):
        x_i = get_x_given_y(newX_train, Y_train, i)
        cov[i] = np.cov(np.array(x_i).T)
        cov[i] = np.add(cov[i], 0.3 * np.identity(cov[i].shape[0]))

    maxP = [-9999999.99 for i in range(testSize)]
    argMaxP = [None for i in range(testSize)]

    # Predict Y for the testing data
    for i in range(10):
        detSigma = nlin.det(cov[i])
        invSigma = nlin.inv(cov[i])
        yProb = pY(Y_train, i)
        for j in range(testSize):
            x = get_transformed_data_vector(testX[j], i, dropList)
            d = x.shape[0]
            
            exp = -1/2 * (np.dot(np.dot(x.T, invSigma), x))
            p = -d/2 * math.log(2 * math.pi) - 1/2 * math.log(detSigma) + exp + math.log(yProb)
            
            if maxP[j] < p:
                maxP[j] = p
                argMaxP[j] = i

    accuracy = np.mean(argMaxP==testY)
    print(accuracy)