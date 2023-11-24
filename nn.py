import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from keras.utils import to_categorical
import pickle
import matplotlib.pyplot as plt

# Load CIFAR-10 test batch
def load_cifar10_test_batch(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data[b'data'], data[b'labels']

# Load CIFAR-10 training and test data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Reshape and normalize the data
train_images = train_images.reshape((len(train_images), 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0
test_images = test_images.reshape((len(test_images), 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Build the MLP model
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)
