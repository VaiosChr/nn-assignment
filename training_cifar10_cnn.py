import neptune
import tensorflow as tf
from keras import layers, models
from keras.datasets import cifar10 as cifar10
import matplotlib.pyplot as plt
import time

# Define parameters
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMDQwYzVmYy01NTU5LTQ2NmMtOTQ3Ny00OTc5NWUyZjA1NmQifQ=="
NEPTUNE_PROJECT = "vaioschr/NN-DL-assignment"
EPOCHS = 10

# Load CIFAR-10 training and test data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Initialize Neptune
run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

# Configure the model and it's layers
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

start_time = time.time()

# Train the model
history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels))

end_time = time.time()

# Log Hyperparameters
run['parameters'] = {
    'epochs': EPOCHS,
    'optimizer': 'adam',
    'loss': 'SparseCategoricalCrossentropy',
    'duration': end_time - start_time,
}

# Log Model Summary
run['model_summary'] = str(model.summary())

# Log Training Metrics
for epoch in range(10):
    run['train/loss'].log(history.history['loss'][epoch])
    run['train/accuracy'].log(history.history['accuracy'][epoch])
    run['val/loss'].log(history.history['val_loss'][epoch])
    run['val/accuracy'].log(history.history['val_accuracy'][epoch])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

run['test_loss'] = test_loss
run['test_accuracy'] = test_acc

# Complete Neptune Experiment
run.stop()

print(test_acc)