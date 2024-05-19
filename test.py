import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model


# Function to resize and invert colors of images from a directory
def load_custom_digits(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('L')  # Open as grayscale
            image = ImageOps.invert(image)  # Invert colors
            image = image.resize((28, 28))  # Resize to 28x28 pixels
            image_array = img_to_array(image).astype('float32') / 255.0
            images.append(image_array.flatten())  # Flatten the image to 1D array
    return np.array(images)


# Directory path where your scanned digits are stored
custom_digits_dir = "/home/space/projects/digit-recognition/digits"

# Load custom digit images (resized, colors inverted, and flattened)
X_custom_digits = load_custom_digits(custom_digits_dir)

# Load the trained model
model_path = "/home/space/projects/digit-recognition/results/keras_mnist.h5"
mnist_model = load_model(model_path)

# Predictions on the custom digit images
predicted_classes = np.argmax(mnist_model.predict(X_custom_digits), axis=-1)

# Display the predictions
for i, predicted_class in enumerate(predicted_classes):
    plt.imshow(X_custom_digits[i].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted Digit: {}".format(predicted_class))
    plt.xticks([])
    plt.yticks([])
    plt.show()
