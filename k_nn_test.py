# program to compare the performance of k-Nearest 
# Neighbors (kNN) and Nearest Centroid classifiers on the CIFAR-10 dataset

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.datasets import cifar10

# load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# reshape the data for kNN and Nearest Centroid
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# train and evaluate k-Nearest Neighbors (kNN) classifier, with k=1
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train, np.ravel(y_train, order='C'))
y_pred_knn_1 = knn_1.predict(X_test)
accuracy_knn_1 = accuracy_score(y_test, y_pred_knn_1)
confusion_matrix_knn_1 = confusion_matrix(y_test, y_pred_knn_1)

# train and evaluate k-Nearest Neighbors (kNN) classifier, with k=3
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, np.ravel(y_train, order='C'))
y_pred_knn_3 = knn_3.predict(X_test)
accuracy_knn_3 = accuracy_score(y_test, y_pred_knn_3)
confusion_matrix_knn_3 = confusion_matrix(y_test, y_pred_knn_3)

# train and evaluate Nearest Centroid classifier
nc = NearestCentroid()
nc.fit(X_train, np.ravel(y_train, order='C'))
y_pred_nc = nc.predict(X_test)
accuracy_nc = accuracy_score(y_test, y_pred_nc)
confusion_matrix_nc = confusion_matrix(y_test, y_pred_nc)

# print the results
# classifier accuracy
print(f'kNN (N=1) Classifier Accuracy:\t\t{accuracy_knn_1:.4f}')
print(f'kNN (N=3) Classifier Accuracy:\t\t{accuracy_knn_3:.4f}')
print(f'Nearest Centroid Classifier Accuracy:\t{accuracy_nc:.4f}')

# confusion matrix
print(f'kNN (N=1) Classifier Confusion Matrix:\n{confusion_matrix_knn_1}')
print(f'kNN (N=3) Classifier Confusion Matrix:\n{confusion_matrix_knn_3}')
print(f'Nearest Centroid Classifier Confusion Matrix:\n{confusion_matrix_nc}')