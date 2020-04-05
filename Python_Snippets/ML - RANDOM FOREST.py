
'''

--- Further development of Decision Tree algorithm (each 'tree' is a decision)

Goal: Build multiple decision trees
The 'random forest' is decision trees built on random sampling (for large datasets)
Each 'Tree' will come up with its own decision, and the final decision can be a 'majority vote'

'''

#############################################################################
# Load / Check Data
#############################################################################

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_digits
digits = load_digits()

dir(digits)

plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])

# create a dataframe
df = pd.DataFrame(digits.data)
df.head()

# these map to the target, so append it in your dataframe (target is the actual number the images represent)
# columns 0-63 (64 bit) is the binary data for the image itself
df['target']=digits.target
df.head()

# Split data into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)



#############################################################################
# Load / Train Model and check predictions
#############################################################################

### (RandomForest is a collection of multiple models, hence it is in the "ensemble" library)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Check parameters in the above output and can play with them
model.score(X_test,y_test)

# Tune the model:
model = RandomForestClassifier(n_estimators=50)
# Retrain
model.fit(X_train, y_train)
# Recheck score
model.score(X_test,y_test)

# Plot confusion matrix to check predictions
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
