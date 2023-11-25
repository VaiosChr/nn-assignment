import neptune
import tensorflow as tf
from keras import layers, models
from keras.datasets import cifar10 as cifar10
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

EPOCHS = 10

# Initialize the neptune environment
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMDQwYzVmYy01NTU5LTQ2NmMtOTQ3Ny00OTc5NWUyZjA1NmQifQ=="
NEPTUNE_PROJECT = "vaioschr/NN-DL-assignment"
run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Reshape images to 1D array
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Normalize pixel values to between 0 and 1
train_images_flat = train_images_flat.astype('float32') / 255.0
test_images_flat = test_images_flat.astype('float32') / 255.0

# Apply PCA
n_components = 100  # Adjust the number of components as needed
pca = PCA(n_components=n_components)
train_images_pca = pca.fit_transform(train_images_flat)
test_images_pca = pca.transform(test_images_flat)

# Reconstruct the images
train_images_reconstructed = pca.inverse_transform(train_images_pca)
test_images_reconstructed = pca.inverse_transform(test_images_pca)

# Reshape the reconstructed images to original shape
train_images_reconstructed = train_images_reconstructed.reshape(train_images.shape)
test_images_reconstructed = test_images_reconstructed.reshape(test_images.shape)

# Initialize the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

# Train the model with the transformed data
history = model.fit(train_images_reconstructed, train_labels, epochs=EPOCHS, validation_data=(test_images_reconstructed, test_labels))

# Log Hyperparameters
params = {
    "max_epochs": EPOCHS,
    "optimizer": "Adam",
}
run["parameters"] = params

# Log Model Summary
run['model_summary'] = str(model.summary())

# Log Training Metrics
for epoch in range(10):
    run['train/loss'].log(history.history['loss'][epoch])
    run['train/accuracy'].log(history.history['accuracy'][epoch])
    run['val/loss'].log(history.history['val_loss'][epoch])
    run['val/accuracy'].log(history.history['val_accuracy'][epoch])

test_loss, test_acc = model.evaluate(test_images_reconstructed, test_labels, verbose=2)

run['test_loss'] = test_loss
run['test_accuracy'] = test_acc

# Complete Neptune Experiment
run.stop()

print(test_acc)