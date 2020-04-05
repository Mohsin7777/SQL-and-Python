


###################################################
# Using Joblib
# For Models based on large numpy arrays
''' deprecation warning for joblib within sklearn
but "conda install joblib" yields some warning, so check that'''
###################################################

# a trained model:
reg = linear_model.LinearRegression()
reg.fit(TRAIN_df[['area']],TRAIN_df.price)

# Save model 'reg' to a file named 'model_joblib'
from sklearn.externals import joblib
joblib.dump(reg, 'model_joblib')

# Load the saved model as 'mp'
mp = joblib.load('model_joblib')




###################################################
# Using Pickle
# For models NOT based on large numpy arrays
###################################################

# Save the trained model to a file named "model_pickle"
# create a binary file in working directory
## contianing the linear_regression() trained model "reg"
import pickle
with open('model_pickle','wb') as f:
    pickle.dump(reg,f)

####### Load model from file (and call the model "mp")
with open('model_pickle','rb') as f:
    mp = pickle.load(f)


