


##########################################
# First do something Manually:
##########################################

### import/prep the rest of the data
import pandas as pd
df = pd.read_csv("spam.csv")
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)


##########################################
# Without Pipeline:
#(1) CountVectorization + (2) Model Fitting
# Do all your transformations before the actual fitting last

# (1) Convert the sentences in 'messages' to numeric, using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
v= CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

# (2) Train the Model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)


####################################################################
########## Instead of above, you can Create Pipeline
####################################################################

# Create the pipeline
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Now you can use the pipeline to perform both the vectorization and the fitting in one step:
clf.fit(X_train,y_train)


# check score again
clf.score(X_test,y_test)
