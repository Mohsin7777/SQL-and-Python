
'''
Check notes in binary logistic regression script for basics
'''

###################################################################
# Load and check the data
###################################################################
## Identify what numerical digit an image maps to

import matplotlib.pyplot as plt

# Load an Example Dataset for images
from sklearn.datasets import load_digits
digits = load_digits()

# show what fields the dataset contains: ('data' has the binary data of images, 'images' has the actual picture)
dir(digits)
# images are represented as 1 dimensional arrays (64 elements, 8x8):
digits.data[0]
# show the first 3 images:
plt.gray()
for i in range(3):
    plt.matshow(digits.images[i])
# see what the 'target' field contains, print the first 5
digits.target[0:5]

#### we will use 'data' and 'target' to train our model
## 'data' contains the numerical image representation
## 'target' contains the correct value
## (the image is the actual image picture, dont need it to train, use numerical)

### split data (!!dont use this SPLIT script for that for real stuff!!)
from sklearn.model_selection import train_test_split
## X variable is already a 1 dimensional array so dont need double brackets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.8)


###################################################################
# Load and Train the model
###################################################################
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

#### check score:
model.score(X_test, y_test)

# pick a random image and see what the model predicts
plt.matshow(digits.images[67])
# need to supply a multidimensional array into predict function [[]]
model.predict([digits.data[67]])

# check where your model is failing (see outliers)
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
# supply the truth versus prediction:
cm = confusion_matrix(y_test, y_predicted)


# visualize the confusion matrix ("cm") result, using seaborn's heatmap
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# This heatmap shows how many times the truth deviated from your model, for each value
# The value on the tile is the # of times the event occured, and the axis shows the truth/prediction values