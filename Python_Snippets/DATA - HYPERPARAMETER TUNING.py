

'''

GridSearchCV and RandomizedSearchCV
- These are used to quickly test the parameters of a model to find the best fit
- They also use KFold Cross Validation (so no need to split train/test)


1) GridSearchCV
Very thorough, but can be very computationally intensive, as it checks all possible permutations

2) RandomizedSearchCV
Quicker, because you can define the number of iterations

'''



#### Import dataset iris
import pandas as pd
from sklearn import datasets

# assuming all prep work is done, as it is for this inbuilt dataset
iris = datasets.load_iris()


#######################################################################################
# 1) Hypertuning with GridSearchSV
#######################################################################################

# Selected model SVM *(assuming you've picked this model after testing against other models)
from sklearn.svm import SVC

# Import Grid Search
from sklearn.model_selection import GridSearchCV

# First define your classifier model (clf)
# Then define your 'parameter grid', values of the parameters you want to loop through (and any you want to keep static e.g. gamma)
# (Keeping gamma static here to avoid warnings, but you can enter tuning values for that too)
# NOTE: GridSearchCV uses Kfold cross validation, so define the number of folds at the end too
# Keep the internal score=False, use the model's native scoring
clf = GridSearchCV(SVC(gamma='auto'), {
    'C': [1,10,20]
    ,'kernel': ['rbf','linear']
}, cv=5, return_train_score=False)

## Then train the model using GridSearchCV ('data' has the X variables, and 'target' as the y variable)
clf.fit(iris.data,iris.target)

# Get results
clf.cv_results_

# Import results into a dataframe for easy viewing
df = pd.DataFrame(clf.cv_results_)

# The Results will be pre-ordered and ranked from best to worst
df

# Most important result columns:
df[['param_C','param_kernel','mean_test_score','std_test_score','rank_test_score']]

## Interpreting results - Above, it shows that the first 3 combinations of parameters all yield a 98% accuracy,
# so any of those combos should be used
# Quick check of which are the best parameters:
clf.best_params_

# Other options for results that are provided by GridSearch
dir(clf)



#######################################################################################
# 2) Hypertuning with RandomizedSearchCV
#######################################################################################

# Key difference here is the "n_iter" parameter, which defines the limit for iterations
# NOTE: exact same parameters to check as previous GridSearchCV, but this wont run all iterations at the same time

from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(SVC(gamma='auto'), {
    'C': [1,10,20]
    ,'kernel': ['rbf','linear']
},
cv=5
, return_train_score=False
,n_iter=2
)

## Then train the model
rs.fit(iris.data,iris.target)
# Save the results:
df = pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score','std_test_score','rank_test_score']]
# NOTE: Every time you run this code block the parameters will change randomly
# Useful in case you have limited computing power
# Eventually, it will go through all the iterations as GridSearchCV, but not at the same time

# Print results after each run and note the down:
df

