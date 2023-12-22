import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Load CIFAR-10 dataset
# Assume you have loaded CIFAR-10 data into X and y
# X should be a 2D array where each row is a flattened image
# y should be an array of corresponding class labels
# Make sure to use only the first two classes (e.g., classes 0 and 1)

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

# Define the SVM model
svm_model = SVC()

# Define the parameter grid for grid search
param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5]
}

# Create a grid search object
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(x_train, y_train.ravel())

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model from grid search
best_svm_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm_model.predict(x_test)

# Evaluate the performance of the best model
accuracy = svm_model.score(x_test, y_test.ravel())
print("Accuracy:", accuracy)
