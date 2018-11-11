'''
Trains a neural network model to classify images of clothing.

Uses Fashion-MNIST - a training set of 60,000 examples and a test set of 10,000 examples.
Each example is a 28x28 grayscale image, associated with a label from 10 classes.
'''

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# The two statements below are required to fix the OMP: Error #15
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# endfix


print(tf.__version__)

# Access the Fashion MNIST directly from TensorFlow
fashion_mnist = keras.datasets.fashion_mnist

# The train_images and train_labels arrays are the training set—the data the model uses to learn.
# The model is tested against the test set, the test_images, and test_labels arrays.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Map each image to a single label, starting from 0 - T-shirt/top

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Exploring data:")
print("Items and dimensions: ", train_images.shape)
print("Total items:", len(train_images))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# chaining together simple layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # transforms the format of the images from a 2d-array to a 1d-array of 28 * 28 = 784 pixels
    keras.layers.Dense(128, activation=tf.nn.relu), # first Dense layer has 128 nodes (or neurons)
    keras.layers.Dense(10, activation=tf.nn.softmax) # layer is a 10-node softmax layer. Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
])

# compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), # This is how the model is updated based on the data it sees and its loss function.
              loss='sparse_categorical_crossentropy', # Loss function measures how accurate the model is during training.
              metrics=['accuracy']) # Used to monitor the training and testing steps.

# Train the model
# Feed the training data to the model - "train_images" and "train_labels" arrays
# The model learns to associate images and labels.
# Ask the model to make predictions about a test set "test_images" array.
# Verify that the predictions match the labels from the test_labels array.

model.fit(train_images, train_labels, epochs=5)

# Compare how the model performs on the test dataset:

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Make predictions

predictions = model.predict(test_images)
# np.argmax(predictions[0]) # which label has the highest confidence value
# test_labels[0]


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)