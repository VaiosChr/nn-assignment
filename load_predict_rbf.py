from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import random
from matplotlib import pyplot as plt

class_names = ['automobile', 'airplane']

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

# Load the saved model
loaded_model = load_model('rbf_model')

for i in range(20):
    index = random.randint(0, len(test_images) - 1)
    img = test_images[index]
    true_label = class_names[int(np.argmax(test_labels[index]))]

    # Flatten the image to match the model's input shape
    img_flattened = img.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(img_flattened)

    # Get the predicted class
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_names[predicted_class_index]

    plt.imshow(img)
    plt.title(f"Actual: {true_label}, Predicted: {predicted_class_label}")
    plt.show()
