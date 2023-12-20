from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
import numpy as np

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

# create a SVM classifier
clf = svm.SVC()

# train the classifier
clf.fit(x_train, y_train)

# test the classifier
score = clf.score(x_test, y_test)

print("Accuracy:", score)
