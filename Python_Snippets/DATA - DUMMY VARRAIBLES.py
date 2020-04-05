'''

NOTE: Do this in SQL (before importing into python)

Here's the problem: There are 2 types of Categorical Variables:
1) Nominal Variables = There is no order (e.g. names of towns)
2) Oridinal Variables = There is an order (e.g. income)

Since we have to convert text columns to numbers, before feeding the data into a ML model,
Dealing with Nominal Variables requires care, otherwise a numerical order will be assumed by the algorithm

Note: Watch out for the "Dummy Variable Trap"
In Python, automatically generating dummy variables will create one redundant column
>>>>>  you only need 4 columns for 5 towns (all zeros will indicate the 5th town)
(Some ML models can handle this, but its better to take care of this in pre-processing)
So if doing this in SQL, use 4 CASE statements for 5 variables etc.

'''



# Import dataset, the column "towns" contains the nominal-categorical variable (names of towns)
import pandas as pd
df = pd.read_csv("homeprices_onehotenc.csv")



##############################
# Dummy Variables
##############################
# Create Binary Values for Each town (1/0) so there is NO numerical order
# Basically, a case statement for each value in the column
# For each town, this creates a seperate binary 1/0 column
dummies = pd.get_dummies(df.town)

# Combine this new array with the original dataframe
merged = pd.concat([df,dummies], axis='columns')


###### Avoiding "DUMMY VARIABLE TRAP" (+ dropping the now redundant "town" column)
###### Pick one of the town dummies and drop it, as well as the text based "town" column itself
final = merged.drop(["town",'west windsor'], axis='columns')

##### train a linear regression model:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = final.drop(['price'],axis='columns')
y = final.price
model.fit(X,y)

##### check prediction of a value, supplying the binary "1" in the relevant field
### Order in the X table is: area, monroe township, robinsville
model.predict([[2800,0,1]])




####################### Alternative:
####################### One Hot Encoder (does the same thing, in a more confusing way)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dfle = df
# generate an array of dummies, and assign it back to your dataframe
dfle.town = le.fit_transform(dfle.town)

# define your X variables, but with .values, because you want a 2 dimensional array
X = df[['town','area']].values

y = dfle.price

from sklearn.preprocessing import OneHotEncoder

# Define the OHE function by telling it that the 0th columnn will be the categorical column in the array
ohe = OneHotEncoder(categorical_features=[0])

X = ohe.fit_transform(X).toarray()

# Take care of dummy variable trap:
# Take all the columns from index 1 onwards (drop 0th column)
X = X[:,1:]

####### from this point on, proceed as above with linear regression