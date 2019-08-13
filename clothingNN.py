import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# Load in data and split it into training data and testing data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Create list to define labels (give a name to a number)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Divide each pixel value by 255 to reduce data size
train_images = train_images/255.0
test_images = test_images/255.0

# Define architecture and layers for model
# Sequential() meand we are defining the layers sequentially
model = keras.Sequential([
    # First layer is input layer (which will be flattened)
    keras.layers.Flatten(input_shape=(28, 28)),
    # Second layer is a dense layer (fully connected) and we use rectify linear unit
    # as activation function
    keras.layers.Dense(128, activation="relu"),

    # Output layer is a dense layer
    # (softmax will tell it to pick different probabilities for outputs)
    keras.layers.Dense(10, activation="softmax")
])


# Set parameters for model
# Optimizer is how your model iteratively updates its weights
# Loss function is how your model evaluates itself
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
# "Epochs" = How many times will the model see this data
model.fit(train_images, train_labels, epochs=5)

# Evaluate model output using test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Use model to predict
prediction = model.predict(test_images)

# Loop through a few images in test data and show input and output
# Using matplotlib to output predictions
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()
