

###################################################
# DECISION TREE
'''
For datasets too messy (high entropy) for logistic regression

Entropy in data is defined as randomness
The ordering/splitting of data is very important here
Some splits will have less entropy than others
>>> some ways to order data will yield more 'information gain', less randomness

---> Two algorithms that can be used are 'Gini' and 'Entropy'
by default, DecisionTreeClassifier() selects one of the models automatically, but can define it as a parameter

'''
###################################################


### Note: Not covered in this example:  Test/Train Split, and avoiding 'Dummy Variable Trap'

import pandas as pd
# Load data
df = pd.read_csv("salaries.csv")
df.head()


# Divide into Target and Independent Variables
# input is the independent variables (all except the last 'salary..' column)
inputs = df.drop('salary_more_then_100k',axis='columns')
# target is the independent variable:
target = df['salary_more_then_100k']


# Need to convert the labels into numbers for ML algorithm
# !! Do this in SQL !!
from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

# create your numerical columns_n, in inputs
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_company.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])
# drop the label columns
inputs_n = inputs.drop(['company','job','degree'], axis='columns')
# should also drop the redundant numeric column (to avoid 'dummy trap') but the tutorial doesnt do that here


# Import and train the Decision Tree model
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,target)

# check score (will be 100% here cuz you trained it on the model your testing against, and its very simple data)
# Should be testing on TESTING dataset !! not the TRAINING !!
model.score(inputs_n,target)

# predict for google, sales exec, masters:
model.predict([[2,2,1]])