

'''
!!! NOTE !!! Hold-out (TRAIN TEST SPLIT) vs. Cross-validation

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







###############################################################################
# Create a TEST/TRAIN sets


# NOTE: Avoid using random.permutation() type functions and saving result
# If the dataset is updated/refreshed, the sets will be mixed up
# For these functions, if needed, see the book for code
# (p86 and 88 in PDF - Hands on machine learning with scikit)

###############################################################################



########################
# Method 1) Customized Random Split
# Only acceptable for very large datasets, especially relative to # of dimensions
########################
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

# CREATE THE FUNCTION:
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# USE THE FUNCTION in 1 of 2 ways:

# OPTION A) IF the dataset does not have an identifier column:
# The simplest solution is to use the row index as the ID
''' !! NOTE: need make sure that any new data is appended to the end of the dataset 
and that no row ever gets deleted !! '''

# adds an `index` column:
housing_with_id = housing.reset_index()
# Apply Function:
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# OPTION B)
'''
you can try to use the most stable features to build a unique identifier. 
For example, a district’s latitude and longitude:
'''
# MAKE A UNIQUE ID BY COMBINING COLUMNS (E.G. LATITUDE AND LONGITUDE):
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# Apply function:
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")




########################
# Method 2) Stratified Sampling
########################

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



########################
# Method 3) Use built in function (avoid this due to reasons listed at the top)
########################

# for tables "X" and "y":
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=10)

# the last "random_state" parameter makes the set persistent (but only as long as dataset stays the same)

