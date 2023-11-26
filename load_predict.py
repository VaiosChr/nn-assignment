from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import random
from PIL import Image

# Load the CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize the data
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Load the model architecture
loaded_model = load_model('cifar10_model.h5')

# Compile the model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

for i in range(20):
    index = random.randint(0, 10000)
    img = test_images[index]
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = loaded_model.predict(img_array)

    # Get the predicted and actual classes
    predicted_class = np.argmax(prediction)
    actual_class = test_labels[index][0]

    # Print the actual and predicted classes
    print(f"Actual Class: {actual_class}, Predicted Class: {predicted_class}")

    # Show the image
    img = image.array_to_img(img)

    # Save the image to Downloads folder as cifar10_{i}.png
    img.save(f"C:\\Users\\vallo\\Downloads\\cifar10_{i}.png")