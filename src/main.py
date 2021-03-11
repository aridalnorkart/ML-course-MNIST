"""
## Setup
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

"""
## Prepare the data
We use numpy to get the data ready for the ML model.
More about numpy: https://numpy.org/doc/stable/user/absolute_beginners.html
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
This section defines the model. Try to see what happens to the training and validation accuracies when you change 
the number of layers and/or the number of filters in each layer. 

The number of filters is changed by changing the first parameter in layers.Conv2D(numberOfFilters). Try 8, 16, 32 or 64
More about Convolution layers: https://keras.io/api/layers/convolution_layers/convolution2d/

Another parameter that can be changed is the activation. Try with "relu" or "sigmoid". 
More activation functions: https://keras.io/api/layers/activations/
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(8, kernel_size=(3, 3), activation="sigmoid"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        #layers.Conv2D(8, kernel_size=(3, 3), activation="sigmoid"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 15

# Try with different optimizers, "sgd" or "adam"
# More about optimizers: https://keras.io/api/optimizers/
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
