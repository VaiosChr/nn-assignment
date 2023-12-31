import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from keras.utils import to_categorical
from keras.datasets import cifar10
import time
import neptune

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Select the first two classes (airplane and automobile)
class_indices = np.where((train_labels == 0) | (train_labels == 1))[0]
train_images = train_images[class_indices]
train_labels = train_labels[class_indices]

class_indices = np.where((test_labels == 0) | (test_labels == 1))[0]
test_images = test_images[class_indices]
test_labels = test_labels[class_indices]

# Preprocess the data
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Flatten the images
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Create grid-based centers
NUM_CENTERS = 100
kmeans = KMeans(n_clusters=NUM_CENTERS, random_state=42).fit(train_images)
centers = kmeans.cluster_centers_

# Define a custom layer to compute RBF distances
class RBFLayer(layers.Layer):
    def __init__(self, centers, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.centers = tf.constant(centers, dtype=tf.float32)

    def call(self, inputs):
        # Compute distances to centers using map_fn
        distances = tf.map_fn(lambda x: tf.norm(x - self.centers, axis=1), inputs, dtype=tf.float32)
        return distances

# Build the RBF Neural Network
model = models.Sequential()
model.add(RBFLayer(centers))
model.add(layers.Dense(2, activation='softmax'))  # Output layer with 2 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=60, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the model on the test set
accuracy = model.evaluate(test_images, test_labels)[1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('rbf_model')