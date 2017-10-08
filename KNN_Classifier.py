import scipy.io
import numpy as np
from numpy.linalg import *

def euclidean_distance(x1, x2):
    no_of_features = x1.shape[0]
    sum = 0
    for i in range(0, int(no_of_features)):
        sum = sum + np.square(x1[i] - x2[i])

    euc_distance = np.sqrt(sum)
    return euc_distance

mat_contents = scipy.io.loadmat('hw1data.mat')
X = mat_contents['X'].astype(np.float64)
Y = mat_contents['Y'].astype(np.float64)

# pruning the features that aren't varying across training samples
features_to_prune = []
for j in range(0, 784):
    test = 0
    for i in range(0, 10000):
        if(X[i, j])!= 0:
            test = 1
    if test == 0:
        features_to_prune.append(j)

X = np.delete(X, features_to_prune, axis=1)

# taking 2/3rd of the data as training data and rest of it as the test data
train_proportion =7000
test_proportion = 3000
Xtrain = X[0:train_proportion, :]
Xtest = X[train_proportion:10000, :]
Ytrain = Y[0:train_proportion, 0]
Ytest = Y[train_proportion:10000, 0]

# 2 if euclidean, 1 if manhattan
order = 2

# this can be modified to see which value of k gives best performance
k = 3

expected_obtained_outputs = np.empty([test_proportion, 2])
for i in range(0, test_proportion):
    expected_obtained_outputs[i, 0] = Ytest[i]
    input_image_vector = Xtest[i, :]
    distances = [0]*train_proportion
    for j in range(0, train_proportion):
        distances[j] = euclidean_distance(Xtrain[j, :], input_image_vector)
        # The below is to be used while recording performance
        # distances[j] = norm(Xtrain[j, :]-input_image_vector, ord=order)

        # This is to be used when L-inf is being used i.e. max distance
        # distances[j] = max(Xtrain[j, :]-input_image_vector)
        print(j)
    distances = np.array(distances)
    ind = np.argpartition(distances, k)[:k]
    output_array = np.array(Ytrain[ind])
    counts = np.bincount(output_array.astype(int))
    expected_obtained_outputs[i, 1] = int(np.argmax(counts))

no_of_misses = 0
no_of_hits = 0
for i in range(0, test_proportion):
    if expected_obtained_outputs[i, 0] == expected_obtained_outputs[i, 1]:
        no_of_hits = no_of_hits + 1
    else:
        no_of_misses = no_of_misses + 1

print("accuracy:  " + str(no_of_hits/test_proportion))