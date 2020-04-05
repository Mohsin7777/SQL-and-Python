
###############################################################################
'''

In Machine Learning, SVMs are supervised learning models/algorithms
that analyze data used for classification and regression analysis.

The goal is to differentiate clusters with the proper dividing line
It does that by maximizing the distance between clusters, with respect to the line
The near-by points (to measure distance from the line) are called "Support Vectors"

(In 3D, the "line" is a "Plane", and for higher dimensions its an N-Dimensional Hyperplane)

High Gamma = Consider only close-in support vectors
Low Gamma = Consider far away support vectors also

High Regularization (C) = over fitting (as needed)
Low Regularization (C) = under fitting (as needed)

Kernel = Transformation of data, adding an extra dimension for drawing decision boundary (if needed)

'''
###############################################################################



###############################################################################
# Load and check the data:
###############################################################################
## Using the sklearn Iris dataset:
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import load_iris
iris = load_iris()

# check all the subsets in Iris
dir(iris)
# ('data' contains the data)
# check the dimension names in feature names:
iris.feature_names

# convert into dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

# Append the target variable
df['target'] = iris.target
df.head()

# append flower name, using reference, using lambda function
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

# Visialize
# Create 3 dataframes (seperate each species of petals)
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

# Draw scatter plot for sepal
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='+')

# Draw scatter plot for petal
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='+')

###############################################################################
# The SVM Model:
###############################################################################
# Data looks pretty clean (clearly seperated)
# Prepare data:
# drop dependent/target variables
X = df.drop(['target','flower_name'], axis='columns')
# y is the numeric target column (name not needed)
y = df.target

# split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

### import and train the SVM model
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

# Test the model on the TEST set
model.score(X_test, y_test)

# Try changing the C parameter (regularization) and Gamma and Kernel
# Check documentation to see all the parameters and play around with them
model = SVC(C=10, gamma=10, kernel ='linear')
# Train again
model.fit(X_train, y_train)
# Test again
model.score(X_test, y_test)
