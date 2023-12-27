from sklearn import svm
from sklearn import datasets
import joblib
import matplotlib.pyplot as plt

# Example: load a dataset (replace this with your data)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Load the saved model for testing
loaded_model = joblib.load('svm_model.pkl')

# Filter images for classes 0 and 1
class_indices = np.where((y_test == 0) | (y_test == 1))[0]

# Get 10 random indices from classes 0 and 1
random_indices = np.random.choice(class_indices, size=100, replace=False)

# Extract random images and their corresponding labels
random_images = x_test[random_indices]
true_labels = y_test[random_indices]

# Flatten the images for compatibility with SVM model
random_images_flatten = random_images.reshape(len(random_images), -1)

# Make predictions using the loaded model
predicted_labels = loaded_model.predict(random_images_flatten)

correct = 0
incorrect = 0

string_labels = ['airplane', 'automobile']

# Display the results
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(random_images[i])
    plt.title(f"True: {int(true_labels[i])}\nPred: {predicted_labels[i]}")
    plt.axis('off')

plt.show()