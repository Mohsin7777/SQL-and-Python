

'''

For linear results (continous values for output)
output layer's activation = 'linear'
Loss Function: Mean Squared Error (MSE).

Binary classification
Make the output layer = 'Sigmoid' for a binary result
>> the result should be in a probability % (chances of value being true)
Loss function: binary_crossentropy

Multi-Class Classification Problem
A problem where you classify an example as belonging to one of more than two classes.
Output Layer Configuration: One node for each class using the softmax activation function.
Loss Function: categorical_crossentropy

'''


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import History  ## for plotting loss metric across epochs
history = History()

model = Sequential()
model.add(Dense(350, activation="relu",input_shape=(X_train9.shape[1],))) #Layer 1 - can also use input_dim = number of columns
model.add(Dense(350,activation="relu"))
# model.add(Dropout(rate = 0.1,seed=100)) # drop out layer to reduce overfitting
model.add(Dense(2,activation = "linear"))

model.compile(loss="mean_squared_error", # 'mse' , 'binary_crossentropy' , 'categorical_crossentropy'
              optimizer="adam", # 'adam' ,  'rmsprop', 'SGD' , ''
              metrics=["mean_absolute_error"])  # 'accuracy' , 'mae'  , 'caetogrical_accuracy'

model.summary()