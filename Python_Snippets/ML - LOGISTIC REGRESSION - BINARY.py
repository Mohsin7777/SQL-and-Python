
#################################################################################
# LOGISTIC REGRESSION  - For Classification Predictions

'''
# Logistic Reg does not predict continuous values (unlike Linear)
# Logistic Reg is used to predict categorical values e.g. Yes/No (unlike Linear)

Classification Types:
1) Binary e.g. 1/0, Yes/No
2) Multiclass Classification e.g. Republican/Democrat/Independent

-- Logistic graphs require "Sigmoidal" Functions: s(z) = 1/(1+e^-z)
--> These functions convert the input to a range between 0 and 1
--> In Log Regression, we are feeding the y=mx+b line into the sigmoidal funtion
'''
#################################################################################


# Task: Predict (Yes/No) if someone will buy insurance or not, based on age

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("insurance_data.csv")

# check scatter plot to see distribution of data
%matplotlib inline
plt.xlabel='area'
plt.ylabel='price'
plt.scatter(df.age,df.bought_insurance, color='red', marker='+')

### split data (!! dont use this quick script below for that!! use the main SPLIT logic in the main script !!)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size=0.9)

### LOAD MODEL:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

## TRAIN MODEL (with TRAIN)
model.fit(X_train,y_train)

# check score (on TEST)
model.score(X_test,y_test)

# Binary prediction for TEST set:
model.predict(X_test)

# Probability prediction for test set in %
model.predict_proba(X_test)


