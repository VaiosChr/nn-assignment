from sklearn import svm
from keras.datasets import cifar10
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
import neptune

# neptune parameters
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMDQwYzVmYy01NTU5LTQ2NmMtOTQ3Ny00OTc5NWUyZjA1NmQifQ=="
NEPTUNE_PROJECT = "vaioschr/NNDL-assignment-2"

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
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['poly', 'sigmoid'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5]
}

for k in param_grid['kernel']:
    for c in param_grid['coef0']:
        model = svm.SVC(coef0 = c, kernel = k)
        run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

        start_time = time.time()

        # train the classifier
        model.fit(x_train, y_train.ravel())

        duration_time = time.time() - start_time

        # test the classifier
        train_score = model.score(x_train, y_train.ravel())
        test_score = model.score(x_test, y_test.ravel())

        # print the results
        run['Training duration'] = duration_time
        run['Training accuracy'] = train_score
        run['Testing accuracy'] = test_score
        run['kernel'] = k
        run['coef0'] = c

        run.stop()