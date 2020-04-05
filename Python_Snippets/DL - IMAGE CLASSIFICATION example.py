
'''
ANN: ARTIFICIAL NEURAL NETWORKS

Simplest form: (Check the graphical cheatsheet for different types)

INPUT LAYER
HIDDEN LAYER(s)
OUTPUT LAYER

THE NEURONS IN THE HIDDEN LAYER FIRE FOR A TRAINED MODEL - BECOMING REINFORCED
THE PATHWAYS FROM THE INPUT TO THE OUTPUT GET REINFORCED BY WEIGHTING (WHICH THE ALGORITHM ADJUSTS)

STEP FUNCTIONS - DISCREET
SIGMOID FUNCTIONS - LIKE LOG, SMOOTH CLASSES
RELU FUNCTIONS - IF VALUE IS < 0 THEN 0, ELSE WHICHEVER VALUE YOU SET

-- KERAS API USES TENSORFLOW, CNTK AND THEANO FOR ITS BACKEND
'''


##################################################################################
# Launch Jupyter through your Tensorflow Environment
##################################################################################
'''
1) Open Navigator
2) Initialize your Tensorflow Environment
3) Launch the Command Prompt, navigate to your directory, and then launch jupyterlab
> cd /D X:\Python_Scripts\Jupyter_Stuff
> jupyter lab
'''


##################################################################################
# Tutorial Project 1) Image Classification with Keras Dataset

# Note: Any dataset downloaded will save to: C:\Users\Administrator\.keras\datasets (or tensorflow_datasets)
##################################################################################

import keras
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Check which backend its using:
keras.backend.backend()

# download dataset
fm = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fm.load_data()

# 60K images, 28x28
X_train.shape
# 10K images
X_test.shape
# check how images are stored numerically (28x28 pixels)
X_train[0]
# matplotlib can show the image
plt.matshow(X_train[0])
# reference provided by the dataset documentation
y_train[0]

####################################
# Normalize the Dataset
# 255 is the max number for color scale , so just divide by 255
X_train = X_train/255
X_test = X_test/255


####################################
# Create a Neural Network Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
model = Sequential()
# Add Input layer - 'Flatten' is converting a 2D array in 1D array, where input is 28x28
model.add(Flatten(input_shape=[28, 28]))
# Add Hidden Layer (can have 2 if needed) - Define number of 'neurons' which computes result
# Number is determined by trial and error
# and their activation function (sigmoid, relu, etc.)
# More neurons = more processing time
model.add(Dense(100, activation="relu"))
# Add Output Layer - defined as 10 here because the y-variable contains 10 possible options
model.add(Dense(10, activation="softmax"))

### Check model
model.summary()

################# Compile model
# The 'loss' function is the same concept as Gradient Descent (Mean Squared Error minimization)
# The optimizer adjust the ages of your neurons
# Metrics determine what kind of metrics to use during the training
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

################# Train model
# the iterations/epochs dont need to be defined (optional)
model.fit(X_train, y_train, epochs=5)

################# TEST ACCURACY
# Second parameter is accuracy (87%), first parameter is "loss" rate
model.evaluate(X_test, y_test)

################
# Result
################
# This will predict the answer for 10K images
yp = model.predict(X_test)
# This shows which index has the highest probability predicted by the model
np.argmax(yp[0])
### Check the reference results (the labels)
class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
class_labels[np.argmax(yp[0])]





























