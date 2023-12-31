from keras.models import load_model
import random
from matplotlib import pyplot as plt

import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10

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

original_shape = (test_images.shape[0], 32, 32, 3)  # Adjust the shape based on your original data
original_images = test_images.reshape(original_shape)

# Step 2: Scale the pixel values back
original_images *= 255.0
original_images = original_images.astype('uint8')

class_names = ['automobile', 'airplane']

# Load the saved model
loaded_model = load_model('rbf_model')

for i in range(20):
  index = random.randint(0, len(test_images))
  img = test_images[index]
  true_label = test_labels[index]

  img = np.expand_dims(img, axis=0)

  # Make prediction
  prediction = loaded_model.predict(img)

  # Get the predicted and actual classes
  predicted_class = np.argmax(prediction)
  actual_class = test_labels[index][0]

  # Print the true label and predicted class
  print(f"True Label: {true_label}")
  print(f"Predicted Class: {predicted_class}")

  # Print the actual and predicted classes
  # print(f"Actual Class: {class_names[actual_class]}, Predicted Class: {class_names[predicted_class]}")
  print("INDEX = ", index)

  plt.imshow(original_images[index])
  plt.title(f"Actual: {class_names[int(actual_class)]}, Predicted: {class_names[int(predicted_class)]}")
  plt.show()