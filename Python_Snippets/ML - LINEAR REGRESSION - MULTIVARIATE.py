



########################################################################
# LINEAR REGRESSION (Multivariate Regression)
########################################################################

# Linear Regression with
#  (y = m1*x1 + m2*x2 +m3*x3 ... +b)
'''
(See above notes for single variable Linear Regression)
NOTE: if 2 independent variables, need 3D graph...
obviously, more variables cant be graphed, and below example has 3 independent variables (4D)
'''

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model  #sklearn also known as scikit

# import training model
# Independent variables: area, bedrooms, age
# Dependent variable: price (house)
reg = linear_model.LinearRegression()
reg.fit(TRAIN_df[['area','bedrooms','age']],TRAIN_df.price)

##################
# Validation checks:

# Check the R^2 score:
reg.score(TRAIN_df[['area', 'bedrooms', 'age']], TRAIN_df.price)
# check slope and intercept
reg.coef_
reg.intercept_
# check a single result:
p = reg.predict([[3000,3,40]])
##################

# To output a test result based on the trained model:
# import/create a dataframe with all the INDEPENDENT VARIABLES you want, and load into model:
TEST_df = pd.read_csv("multivariate_input.csv")
# create new array of output prices (!!input needs to be aligned, same order of columns!!)
result_df = reg.predict(TEST_df)
# bring in this result column into the original dataframe
TEST_df['prices'] = result_df
# output file
TEST_df.to_csv("prediction_multi.csv", index=False)