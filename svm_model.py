from sklearn import svm
from keras.datasets import cifar10
import numpy as np
from sklearn.externals import joblib

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

model = svm.SVC(kernel = 'poly')

# train the classifier
model.fit(x_train, y_train.ravel())
joblib.dump(model, 'svm_model.pkl')

# test the classifier
train_score = model.score(x_train, y_train.ravel())
test_score = model.score(x_test, y_test.ravel())

# print the results
print("Train Accuracy:", train_score)
print("Test Accuracy:", test_score)