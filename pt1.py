#deploy tensorflow

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

#Step 2: Load and Preprocess Data
#The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0–9).

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


#step 3,build the model for training
# Flatten: Our image is a 2D square ( 28×28 ). We unroll it into a single line of  784  pixels.
# Dense Layer: This is where the learning happens. Each neuron looks for patterns.
# ReLU: An activation function that helps the model learn complex, non-linear patterns.
# Softmax: Since we have 10 possible digits, this layer turns the output into probabilities that sum to  1 .

model = models.Sequential([
  layers.Flatten(input_shape=(28, 28)),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

#step 4, evaluate the model

# 1. Pick an index (e.g., 0, 6, or any number up to 9999)
image_index = 5173

# 2. Reshape the image to add a "batch" dimension
# Models expect a list of images, so we turn [28, 28] into [1, 28, 28]
input_image = x_test[image_index:image_index+1]

# 3. Get the prediction
prediction = model.predict(input_image)
predicted_label = prediction.argmax()
actual_label = y_test[image_index]

# 4. Display the result
print(f"Model's Prediction: {predicted_label}")
print(f"Actual Label: {actual_label}")

plt.imshow(x_test[image_index], cmap='gray')
plt.title(f"Predicted: {predicted_label}, Actual: {actual_label}")
plt.show()