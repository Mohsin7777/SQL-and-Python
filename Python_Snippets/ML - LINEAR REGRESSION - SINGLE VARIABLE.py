



########################################################################
# LINEAR REGRESSION - Single Variable
########################################################################
'''
# Require a "Train" and "Test" dataset
# Train = Scattered Data (independent and dependent variables) used to Train the Model to find 'm' and 'b'
# Test = Preset values of (Independent) variables that the result (Dependent) variable is attached to

# draws a line through a scatter plot to provide a linear estimate of all missing values
# What you need to start with is some validated data to 'train' the model on (where both X and Y values are present)

# based on: y = mx + b
(where 'm' & 'b' are unknown, and X & Y are provided by the "Training" dataset)

# Y = dependent variable
# X = independent variable
# m = slope
# b = y intercept

# Once "m" and "b" are discovered by the algorithm, the "Test" Dataset feeds "x" values, to get "y" as the result
'''


################## EXAMPLE 1)

# Linear Regression of house prices versus area(square feet)
# Dependent Variable: Price
# Independent Variable: Area
# Equation: (price = m*area + b)

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model  #sklearn also known as scikit

# Feed the TRAIN model, with scattered but validated data
TRAIN_df = pd.read_csv("homeprices.csv")  #"train" dataset

# first check the scatter plot:
%matplotlib inline
plt.xlabel='area'
plt.ylabel='price'
plt.pyplot.scatter(TRAIN_df.area,TRAIN_df.price, color='red', marker='+')

# Load/'Train' the linear regression model with the independent and dependent variables
reg = linear_model.LinearRegression()
reg.fit(TRAIN_df[['area']],TRAIN_df.price)  #first variable is independent, second is dependent

######################
# Validation:
# Check the R^2 score (get close to 100%)
# ".score" compares the predicted values to the actual y-values
reg.score(TRAIN_df[['area']],TRAIN_df.price)
# check slope and intercept (this is the actual result:)
reg.coef_
reg.intercept_
# check the prediction for single value (input area, output price)
reg.predict([[3300]])
######################

# To output a result based on the trained model, use the TEST set (with preset independent variables, no dependent):
# import/create a TEST dataset with all the INDEPENDENT VARIABLES you want, and load into model:
TEST_df = pd.read_csv("areas.csv")
# create new array of output prices (!! input needs to be aligned, same order of columns!!)
result_df = reg.predict(TEST_df)
# bring in this result into the original dataframe
TEST_df['prices'] = result_df
#output file
TEST_df.to_csv("prediction_single.csv", index=False)

# Plot the resulting line on the original scatter plot (original file still = 'df')
# A) copy original scatter plot:
%matplotlib inline
plt.xlabel='area'
plt.ylabel='price'
plt.pyplot.scatter(TRAIN_df.area,TRAIN_df.price, color='red', marker='+')
# B) Then plot the line: in the place of Y values, use the trained model (which analyzes the original X values)
plt.pyplot.plot(TRAIN_df.area, reg.predict(TRAIN_df[['area']]), color='blue')



################ EXAMPLE 2)
### SPLIT THE TRAINING DATASET INTO "LABEL" (DEPENDENT) AND "PREDICTOR" (INDEPENDENT) VARIABLE SUBSETS
# AND THEN SEE HOW WELL THE MODEL PREDICTS A SELECTED SUBSET OF THE TRAINING "LABELS"
# ... BEFORE USING IT ON THE ACTUAL "TEST" SET

# 1) Train the model:
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# The model will learn the linear relationship between the 'predictors' (in prepared)
# and the median house value (in the Labels set)
lin_reg.fit(housing_prepared, housing_labels)

# 2) Check how well the 'predictors' predict the median value ('labels')
## insert 5 rows from the source table (before cleaning)
some_data = housing.iloc[:5]
## insert 5 rows from the labels
some_labels = housing_labels.iloc[:5]
## transform the unprepared data using the pipeline (can avoid this by uploading pre-prepared data via SQL)
some_data_prepared = full_pipeline.transform(some_data)
## print prediction values for the 5 rows:
print("Predictions:", lin_reg.predict(some_data_prepared))

# Eyeball the result, see the delta between the model's prediction and the actual median value in "labels"

# 3) Check the MSE: Mean Squared Error:
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
# the result in this case is an MSE of $68K which is not that great
# the data is 'underfitting the training data'
# i.e. the features do not provide enough data to predict accurately
# or that the model is not powerful enough
'''
The main ways to fix underfitting:
A) Select a more powerful model
B) Feed the training algorithm with better features
C) Reduce the constraints on the model.
'''