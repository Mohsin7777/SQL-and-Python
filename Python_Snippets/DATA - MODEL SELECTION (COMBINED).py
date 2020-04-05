
'''

--------------------------- USE THIS SCRIPT ABOVE ALL OTHERS ----------------------------------

THIS COMBINES THE FOLLOWING:

-- KFOLD CROSS VALIDATION
-- HYPERPARAMETER TUNING

AND USES THE ABOVE TO TEST DIFFERENT MODELS AT THE SAME TIME
'''


import pandas as pd
from sklearn import datasets

# Import and prepare dataset
iris = datasets.load_iris()

# IMPORT MODELS/CLASSIFIERS THAT YOU WANT TO USE/TEST:
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Import Gridsearch
from sklearn.model_selection import GridSearchCV


# WRITE A JSON object / Python Dictionary
# Define the list of parameters to use with GridSearchCV for each model
model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}

### Write a FOR loop for GridSearchCV, and call the dictionary above
# Also define the # of KFolds and turn off the internal scoring (use the score native to model)

# make an array to contain the scores
scores = []

# Define the loop which will train & test all the models:
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Result will show the best model with its parameters, ranked
df