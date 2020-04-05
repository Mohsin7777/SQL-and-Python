
'''

Basic Conditional Probability:
P(A/B) = Probability of event A, knowing that event B has already occurred

Bayes Theorem: (Adds more parameters)
P(A/B) = [P(B/A)*P(A)]/P(B)

--> Knowing certain events you can find the probability of unknowns
--> It's called "Naive" because assumption is: those other events are independent of each other

3 types of classifiers:
Gaussian = for normal distribution (Bell Curve)
Bernoulli = when all *features* (not targets) are binary
Multinomial = discrete data e.g. movie ratings 1-5

'''


#########################################################################################
# Dataset #1 - Titanic  - Using Gaussian
#########################################################################################

### TITANIC DATASET - PREPROCESSED IN SQL
from sqlalchemy import create_engine
sqlcon = create_engine('mssql+pyodbc://@' + 'GHOST-117\SQLEXPRESS' + '/' + 'MOHSIN' + '?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server')

# Import with index/ID as column1, to keep the order the same:
df = pd.read_sql_query("SELECT DISTINCT PassengerId, Pclass,male,Age,Fare,Survived FROM  MOHSIN.DBO.TITANIC_PREPROCESSED", sqlcon)

# Drop ID column and split X (inputs) , y (target)
# X variables
inputs = df[['Pclass','male','Age','Fare']]
# y variable
target = df[['Survived']]

## split train/test ( !!! use stratified for real dataset of this small size !!!)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)


############ Create the model
############ Using Gaussian NB, when data distribution is 'normal' (bell curve)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
# Train the model
model.fit(X_train,y_train)

model.score(X_test,y_test)

# Shows the probability for each passenger:
# first column for prediction for death, second is probability for surviving
model.predict_proba(X_test[:10])


###################################################################################################
# Dataset #2 - Spam Filter  - Using Multinomial
#
# (and then using Pipeline to automate some of the steps)
###################################################################################################

# Need to convert 2 columns to numbers (spam category is easy, but the message column is complicated)
import pandas as pd
df = pd.read_csv("spam.csv")

# Convert Spam Category to number, using CASE-like function (Lambda):
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()

# Before converting the message column to numeric, split Train/Test to seperate messages
## split train/test ( !!! use stratified for real dataset of this small size !!!)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)

# Now Convert the sentences in 'messages' to numeric using CountVectorizer
# It takes instances of all unique words to create a corpus, and then provides their occurances
from sklearn.feature_extraction.text import CountVectorizer
v= CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3]

# Load/Train
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

# Test - first convert X_test messages into numeric
X_test_count = v.transform(X_test)
model.score(X_test_count,y_test)

# test with example text
emails = [
    'hey dude, whats up?',
    'Get a Free Discount, 20% off!'
]
email_count = v.transform(emails)
model.predict(email_count)

####################################################################
########## Use Pipeline Transformation to automate the v-transform
####################################################################

# First create the pipeline
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Now you can use the pipeline to perform both the vectorization and the fitting in one step:
clf.fit(X_train,y_train)

# check score again
clf.score(X_test,y_test)





