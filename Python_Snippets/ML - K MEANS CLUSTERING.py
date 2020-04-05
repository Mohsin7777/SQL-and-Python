
'''

This is an unsupervised algorirthm = No labels/Target variables are defined
---> Goal: Find an underlying structure in the data without feeding in any targets

Divides data into 'clusters' based on proximity
'k' is a free parameter - number of groups you want to find
---> These are points on a map and algorithm divides based on proximity to them

K-Means is basically tuning the model until it breaks up the data into clusters which make sense
---> 'make sense' = none of the datapoints shift the clusters anymore upon recalculation
To determine the number of k's you will need the 'elbow method', using SSE (sum of squared errors)
---> The 'elbow' point in the SSE graph (looks like a hinge) is the best number of k's

'''

###################################################################################
# Load and Check
# (example below is without any test/train splitting)
###################################################################################

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("income.csv")
# Drop the 'names' columns
df = df.drop(['Name'], axis='columns')
# Rename income column to remove brackets (easier to type over and over again!)
df = df.rename(columns={"Income($)": "Income"})
df.head()

# plot a scatter to see distribution
plt.scatter(df.Age,df.Income)


###################################################################################
# Scale Data
###################################################################################

# Detected Issue: scaling is messed up
# The y axis goes from 40K to 160K (very wide), while X axis is only 27-42 (too narrow)

# Use MinMax scaler to scale the income feature
# It will rescale income to a range between 0 and 1
scaler = MinMaxScaler()
scaler.fit(df[['Income']])
df.Income = scaler.transform(df[['Income']])

# Do the same scalling for Age (between 0 and 1)
scaler.fit(df[['Age']])
df.Age = scaler.transform(df[['Age']])

df.head()


###################################################################################
# 'Elbow' Method to pick the right # of ks (for more complex datasets)
###################################################################################

# Since this is a simple distribution with clear clusters (in 2 dimensions) its easy to see 3 ks are needed
# HOWEVER, use the 'elbow method' to verify (use this for more complex datasets that wont be able to be visualized)
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income']])
    sse.append(km.inertia_)  # this appends the SSE error to each iteration

# plot the elbow plot (result: k=3 at the elbow/hinge point of the graph)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


###################################################################################
# Load/Train model
###################################################################################

# Load model, use 3 clusters:
km = KMeans(n_clusters=3)
# see available parameters (just use default here, check documentation for options)
km

# can fit and predict at the same time with every model, by the way
y_predicted = km.fit_predict(df[['Age','Income']])
y_predicted

######################### Final Result File:
# append new result of mapped clusters  (This is the result)
df['cluster'] = y_predicted

# These are the X,y values of clusters (based on proximity)
# just for info purposes
km.cluster_centers_


# Plot the final results (color code each cluster)
# Need 3 dataframes, one for each cluster
df0 = df[df.cluster==0]
df1 = df[df.cluster==1]
df2 = df[df.cluster==2]

plt.scatter(df0.Age,df0['Income'],color='green', label='income')
plt.scatter(df1.Age,df1['Income'],color='red',label='income')
plt.scatter(df2.Age,df2['Income'],color='black', label='income')

# plot the k 'centroids' (x is the 0th column, y is the 1st column)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.legend()
