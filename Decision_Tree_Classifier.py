import scipy.io
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

mat_contents = scipy.io.loadmat('hw1data.mat')
X = mat_contents['X'].astype(np.float64)
Y = mat_contents['Y'].astype(np.float64)

# the parameter to tune test to train split proportion
test_2_train_split = 0.7

# the below function use ensures that I have a random split between train and test data
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = test_2_train_split, random_state = 200)

# lists of training and test error to finally plot a graph to observe their variation as per the value of K
train_error = list()
test_error = list()

# list of kyperparameter ranging from 1 to 100
K_hypermeter = [i for i in range(1, 101)]

train_error_value = 0
test_error_value = 0
for k in K_hypermeter:
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=k, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    y_pred_test = clf_gini.predict(X_test)
    y_pred_train = clf_gini.predict(X_train)
    train_error_value = 100-accuracy_score(y_train,y_pred_train)*100
    test_error_value = 100-accuracy_score(y_test,y_pred_test)*100
    train_error.append(train_error_value)
    test_error.append(test_error_value)

optimal_k = 0
for i in range(0, 100):
    if test_error[i] == test_error_value:
        print("The optimal value of hypermeter K is: " + str(i+1))
        optimal_k = i
        break

print("The accuracy is: " + str(100-test_error[optimal_k]))

plot1, = plt.plot(K_hypermeter, train_error, 'ro', label='Training Error')
plot2, = plt.plot(K_hypermeter, test_error, 'bo', label='Test Error')
plt.axis([0, 100, 0, 100])
plt.xlabel('Hyperparameter- K')
plt.ylabel('Train/Test Error')
plt.legend(handles=[plot1, plot2])
plt.show()