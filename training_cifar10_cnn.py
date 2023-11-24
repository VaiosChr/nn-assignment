import neptune
import tensorflow as tf
from keras import layers, models
from keras.datasets import cifar10 as cifar10
import matplotlib.pyplot as plt

# Initialize the neptune environment
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMDQwYzVmYy01NTU5LTQ2NmMtOTQ3Ny00OTc5NWUyZjA1NmQifQ=="
NEPTUNE_PROJECT = "vaioschr/NN-DL-assignment"
run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

params = {
    "max_epochs": 10,
    "optimizer": "Adam",
}
run["parameters"] = params

# Load CIFAR-10 training and test data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Log Hyperparameters
run['parameters'] = {'epochs': 10}

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

# plt.show(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# print(test_acc)