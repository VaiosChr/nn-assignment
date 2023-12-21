from sklearn import svm
from keras.datasets import cifar10
import numpy as np
import time
from sklearn.model_selection import GridSearchCV

# load the cifar-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# filter for 'airplane' and 'automobile' classes
train_picks = np.where((y_train == 0) | (y_train == 1))[0]
test_picks = np.where((y_test == 0) | (y_test == 1))[0]

x_train = x_train[train_picks]
y_train = y_train[train_picks]
x_test = x_test[test_picks]
y_test = y_test[test_picks]

# flatter the images for the svm
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5]
}

# create a grid search object
grid = GridSearchCV(svm.SVC(), param_grid, cv=5, verbose=2, n_jobs=-1)

start_time = time.time()

# train the classifier
grid.fit(x_train, y_train.ravel())

duration_time = time.time() - start_time

# get the best parameters
best_params = grid.best_params_

# test the classifier
train_score = grid.score(x_train, y_train.ravel())
test_score = grid.score(x_test, y_test.ravel())

# print the results
print(f"Best parameters: {best_params}")
print(f"Training duration: {duration_time} seconds")
print(f"Training accuracy: {train_score}")
print(f"Testing accuracy: {test_score}\n")

