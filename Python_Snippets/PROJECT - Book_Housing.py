###############################################################################
###############################################################################
# Project 1  - Bay Area Housing Prices

# Your model should learn from this data and be able to predict the median
# housing price in any district, given all the other metrics.
###############################################################################
###############################################################################




###############################################################################
# Load the data
###############################################################################
import pandas as pd
housing = pd.read_csv("book_dataset/housing/housing.csv", thousands=',')

# For downloading / unzipping:
'''
import os
import tarfile
import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handsonml2/
master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
    
housing = load_housing_data()
'''


###############################################################################
# Examine data - initial observations
###############################################################################

housing.head()
housing.info()
housing.describe()

# Plot a histogram for all numerical dimensions in the table automatically, using hist()
#this tells jupyter to use its own backend for matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()  # this is optional, jupyter executes the cell automatically



###############################################################################
# Create a TEST/TRAIN sets
###############################################################################

# Method 1) Random Split (okay for very large datasets, especially relative to # of dimensions)
# NOTE: Avoid using random.permutation() type functions and saving result
###### If the dataset is updated/refreshed, the sets will be mixed up
####### For these functions, if needed, see the book for code (p86 and 88 in PDF)

'''
To have a stable train/test split even after updating the dataset, a common
solution is to use each instance’s identifier to decide whether or not it should
go in the test set (assuming instances have a unique and immutable
identifier). For example, you could compute a hash of each instance’s
identifier and put that instance in the test set if the hash is lower than or equal
to 20% of the maximum hash value. This ensures that the test set will remain
consistent across multiple runs, even if you refresh the dataset. The new test
set will contain 20% of the new instances, but it will not contain any instance
that was previously in the training set.
'''

from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# If the dataset does not have an identifier column:
# The simplest solution is to use the row index as the ID

housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

''' (!!NOTE!!) Alternative:
If you use the row index as a unique identifier, you need to make sure that
new data gets appended to the end of the dataset and that no row ever gets
deleted. If this is not possible, then you can try to use the most stable features
to build a unique identifier. For example, a district’s latitude and longitude are
guaranteed to be stable for a few million years, so you could combine them
into an ID like so:
'''

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")



# Method 2) Non-Random Splitting ("Stratified Sampling")
''' If the dataset isn't large, then random splits may introduce sampling bias
The set has to be representative of the whole population
the population is divided into homogeneous subgroups called
strata, and the right number of instances are sampled from each stratum to
guarantee that the test set is representative of the overall population.
It is important to have a sufficient number of instances in your dataset for each stratum, 
or else the estimate of a stratum’s importance may be biased
This means that you should not have too many strata, and each stratum should be large enough.
'''

# The following code uses the pd.cut() function to create an income category attribute
# with five categories (labeled from 1 to 5): In the dataset 1 = $10,000
# category 1 ranges from 0 to 1.5 (i.e., less than $15,000),
# category 2 from 1.5 to 3, and so on

housing["income_cat"] = pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])

# check histogram:
housing["income_cat"].hist()

# Now split using 'StratifiedShuffleSplit' function, based on created strata "income cat"
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Check the split ratios (also good as a 'group by' to analyze data):
# NOTE: The proportions should be identical to those in the full dataset
## (the set generated using random split was skewed)
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# Now remove the income_cat attribute so that the data is back to original state:
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Copy the train set into your working df
housing = strat_train_set.copy()



###############################################################################
# Visualize to discover the data
###############################################################################

# plot initial graph
housing.plot(kind="scatter", x="longitude", y="latitude")
# Setting alpha to reduced value of 0.1 highlights the high density points
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# s = population (radius of circle)
# c = price (cmap/jet option preset for blue=low, red=high)
housing.plot(
    kind="scatter"
    , x="longitude"
    , y="latitude"
    , alpha=0.4
    , s=housing["population"]/100
    , label="population"
    , figsize=(10,7)  # size of graph
    , c="median_house_value"
    , cmap=plt.get_cmap("jet")
    , colorbar=True
)
plt.legend()



###############################################################################
# Look for Correlations
# (+ note anomalies to fix before loading into ML model)
###############################################################################
'''
linear correlation: if x goes up, y goes up/down

nonlinear relationships: (e.g.) if x is close to 0, then y generally goes up
'''

# Linear correlation for small datasets:
# For small datasets, you can compute the "Pearson's r"/standard correlation coefficient
## between any pair of attributes (linear only)
# Check linear correlation between median value and all other dimensions:
# +1 is perfect positive, -1 is perfect inverse, 0 means no linear correlation
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# focus on a few promising linear correlations using pandas scatter matrix
# all combinations of the selected attributes will be plotted (so limit # of attributes!)
# (the main diagonal set is the value plotted against itself, pandas uses histogram for this set)
# NOTE: for a strong correlation: you will clearly see the upward trend, and the points are not too dispersed
from pandas.plotting import scatter_matrix
attributes = [
    "median_house_value"
    , "median_income"
    , "total_rooms"
    ,"housing_median_age"
]
scatter_matrix(housing[attributes], figsize=(12, 8))


# Most promising is median house value versus median income, focus on this
# Note, artificial anomalies may show up here, e.g. horizontal lines (in this case, "price caps" enforced by the data source
# These can be noted and removed from the data, before feeding into the ML training algorithm
housing.plot(
    kind="scatter"
    , x="median_income"
    , y="median_house_value"
    ,alpha=0.1)


###############################################################################
# Create new relevant dimensions based on observations
###############################################################################
# Create new relevant columns to increase clarity of dataset
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

# Take another look at correlations with new attributes:
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


############################################
# Start with the clean TRAIN set
# Seperate the 'predictor' and the 'label' columns (may require different transformations for each set)
# (e.g. median value (label) should be 'predicted' by the other attributes)

# Removes the 'label' dimension
housing = strat_train_set.drop("median_house_value", axis=1)
# Create another set with this 'label' dimension
housing_labels = strat_train_set["median_house_value"].copy()
#############################################



###############################################################################
# Prepare Data for ML ingestion
###############################################################################

''' --- BELOW DATA-PREP STEPS SHOULD BE DONE IN SQL-STAGE --- !!

>>>> THE ONLY PROBLEM WITH THAT IS FLEXIBILITY 
>>>> E.G. 'TRANSFORMATION PIPELINE' ALLOWS TURNING PARAMETERS ON/OFF
>>>> THIS ALLOWS FLEXIBILITY WHEN USING/TRAINING ML ALGORITHMS
>>>> E.G. USING AVERAGES TO FILL NULL... CAN TURN THIS ON/OFF


#####################
# NOTE: Only 'fit'/train the transformations to the TRAINING set (then apply those to the TEST set later)

# NULL handling, 3 quick options:
housing.dropna(subset=["total_bedrooms"]) # drops the districts with nulls
housing.drop("total_bedrooms", axis=1) # drops the entire column
median = housing["total_bedrooms"].median() # set null to some default value (average in this case)


# use imputer to handle NULL values in this dataset (replaces NULLs with median/average of column)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# drop non-numeric column
housing_num = housing.drop("ocean_proximity", axis=1)
# train to learn averages of all columns
imputer.fit(housing_num)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)

# OneHotEncoder for Text/Category column:
housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

######################
# Custom transformers (p107 pdf): (for cleaning, combining columns)
# These should work seamlessly with transformation pipelines (next step)

# create a class and implement three methods:
# fit() (returning self)
# transform()
# fit_transform()/TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#####################
# Feature Scaling
Machine Learning algorithms don’t
perform well when the input numerical attributes have very different scales.
This is the case for the housing data: the total number of rooms ranges from
about 6 to 39,320, while the median incomes only range from 0 to 15

There are two common ways to get all attributes to have the same scale: 
1) min-max scaling / 'Normalization', using MinMaxScaler()
2) Standardization, using StandardScaler()


#####################
# Tranformation Pipelines:

# Pipeline #1 for Numeric Columns:
(to order and combine all the steps you need)
Order: All but the last estimator must be transformers
StandardScaler() is new here, but imputer and attribs_adder are done above

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)

# Pipeline #2 to handle categorical/text columns
First we import the ColumnTransformer class, next we get the list of
numerical column names and the list of categorical column names, and then
we construct a ColumnTransformer.

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)

'''

###############################################################################
# Begin Machine Learning
###############################################################################

# output of previous cleaning steps on the TRAINING dataset:
# 1) "housing_prepared" (contains all the cleaned "predictor" dimensions)
# 2) "housing_labels" (contains the median value, the 'label' dimension)


########################
# Linear Regression:
########################

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


########################
# Decision Tree Regressor:
########################

# More powerful model than linear regression, this one is non-linear

# Train the model:
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# Check the MSE of the model's predictions:
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

'''
In this case, the MSE is reported at 0.0
which clearly indicates that the model is now overfitted
-- To validate this, need to split the TRAINING data up further (without touching the TEST data)
'''
# Split training set into 10 pieces and then validate the MSE score on each (using cross_val_score)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
#Now the decision-tree-regression scores are really bad

# Compare with Linear Regression score) using the same approach
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores) # Resused defined function above

'''
The Decision Tree Regressor is overfitting so badly its performing worse than Linear Regression
'''




################################
# Random Forest Regressor
################################

'''
Random Forests work by training many Decision Trees on random
subsets of the features, then averaging out their predictions. Building a model
on top of many other models is called Ensemble Learning, and it is often a
great way to push ML algorithms even further
'''

# Load/Train Forest Regressor:
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

# Check MSE:
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# Check cross validation scores:
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
'''
MSE has gone down from $60-70K, to around $20K
'''
# Also check a few sample results by eyeballing them (using sample created in lin_reg step)
print("Predictions:", forest_reg.predict(some_data_prepared))
# versus label:
print("Labels:", list(some_labels))


###############################################################################
# Fine Tuning your Model
###############################################################################
'''
(page 117 pdf) 
If you're using hyperparameters/pipelines (instead of preparing your data in SQL)
You can use automatic methods to check different variations of parameters
>> This allows you to fine tune your model with all combinations of these parameters
>> check which parameters work best and then select those

Below code is fine tuning using Grid Search:
'''

from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3,
4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_estimator_


###############################################################################
# Evaluate your model on the TEST dataset
###############################################################################

final_model = grid_search.best_estimator_

# "strat_test_set" is the TEST dataset created using stratification split initially

# All the independent variables (dropping the dependent variable or 'label')
X_test = strat_test_set.drop("median_house_value", axis=1)
# The dependent variable (or 'label')
y_test = strat_test_set["median_house_value"].copy()
y_test

# Transform the independent dataset using the pipeline
X_test_prepared = full_pipeline.transform(X_test)
# Apply the final model:
final_predictions = final_model.predict(X_test_prepared)

# check the MSE again:
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse

'''
In this case, the MSE for the model applied to the TEST set is $47K
NOTE: Fine-Tuning the model on the Training data too much isn't likely to translate to the TEST dataset

In this California Housing example, 
the final performance of the system is not better than the experts’ price estimates,
which were often off by about 20%,

but it may still be a good idea to launch it, 
especially if this frees up some time for the experts so they can work on
more interesting and productive tasks
'''