
'''
!!! NOTE 1 !!! Hold-out (TRAIN TEST SPLIT) vs. Cross-validation

TRAIN/TEST SPLITS (ADVANTAGES AND DISADVANTAGES:)
The hold-out method is good to use when you have a very large dataset,
you’re on a time crunch, or you are starting to build an initial model in your data science project.
Hold-out, on the other hand, is dependent on just one train-test split.
That makes the hold-out method score dependent on how the data is split into train and test sets.

CROSS VALIDATION (KFOLD) ADVANTAGES/DISADVANTAGES
Cross-validation is usually the preferred method
because it gives your model the opportunity to train on multiple train-test splits.
This gives you a better indication of how well your model will perform on unseen data.
Keep in mind that because cross-validation uses multiple train-test splits,
it takes more computational power and time to run than using the holdout method.

'''


'''
!!! NOTE 2: Use GridSearchCV or RandomizedSearchCV instead of running KFold by itself !!!
>>>>>>>> That combines Hyperparameter Tuning with Cross Validation


Using Cross Validation By itself (without hyperparamter tuning)

Quick and simple To figure out which model to use (based on average score)
>>>>>>>> Divide your set into 'folds' and iteratively train/test different models FOR EACH FOLD
>>>>>>>> Then average the scores of each model

This is easily done using built in cross_validation function
'''


#######################################################################################
# Load All the models you want to train/test + your data
#######################################################################################
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
'''
import warnings
warnings.filterwarnings(action='ignore')
'''
from sklearn.datasets import load_iris
iris = load_iris()



#######################################################################################
# Begin Cross Validation for each model and Average their scores
# (Dont need to split train/test, it creates 'folds' iteratively and automatically)

''' Function parameters:

cross_val_score(Argument1 = model, Argument2 = X, Argument3 = y, cv = number_of_folds/iterations) 

!!! Note: !!!
-- If you dont specify 'cv' it will default to simple KFold (without stratification splitting)
-- When 'cv' parameter is an integer:
>>>>>>>>>>>>>>> if the estimator/model is a classifier and y is either binary or multiclass, StratifiedKFold is used. 
>>>>>>>>>>>>>>> In all other cases, KFold is used.

For other parameters (e.g. groups, scoring etc.) check documentation 
https://scikit-learn.org/stable/modules/cross_validation.html
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
'''
#######################################################################################

########## Logistic Regression
l_scores = cross_val_score(LogisticRegression(), iris.data, iris.target, cv = 3)
# Average of scors:
np.average(l_scores)
# fancy output of model accuracy (avg and standard deviation):
print("Accuracy: %0.2f (+/- %0.2f)" % (l_scores.mean(), l_scores.std() * 2))



########## Decision Tree
d_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv = 3)
np.average(d_scores)

########## Support Vector Machines
s_scores = cross_val_score(SVC(), iris.data, iris.target, cv = 3)
np.average(s_scores)

########## Random Forest
r_scores = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target, cv = 3)
np.average(r_scores)

'''
Should tune parameters for each model too, like in the RandomForest example
Play around with the parameters for each to optimize the models and then select the best one
'''

######## You can manually use K Fold as well, check the notebook "k fold details"
######## to see how the backend works, the notebook has the details



######################################################
# Manual KFold for Neural Networks:
######################################################

#( build and compile the network first)

k = 3
num_val_samples = len(X_train9) // k
num_epochs = 50
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = X_train9[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train2[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [X_train9[:i * num_val_samples],
         X_train9[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train2[:i * num_val_samples],
         y_train2[(i + 1) * num_val_samples:]],
        axis=0)

    model = model
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)