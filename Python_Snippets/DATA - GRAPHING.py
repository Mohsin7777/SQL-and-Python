


# In plotting functions, X variable comes before y variable (by default, unless specifically defined)
# If using Jupyter notebook, use "%matplotlib inline" to use Jupyter's backend
# Also in Jupyter, dont need to use the plt.show() function, as it executes automatically when running cell

# General command for matplotlib: figsize=(20,15) defines dimensions of graph (20 by 15)

########################################################################
# Line Graph
########################################################################
import matplotlib.pyplot as plt
%matplotlib inline
plt.xlabel='area'
plt.ylabel='price'
plt.plot(TRAIN_df.area,TRAIN_df.price, color='red', marker='+')




############
# Detailed example:
x=[0,1,2,3,4]
y=[50,56,45,67,45]

# Labels:
plt.xlabel('Day')
plt.ylabel('Stuff')
plt.title('Random_chart')
plt.plot(x,y,color='green',linewidth=4,linestyle='dotted')

# Condensed command 'marker' (plots '+' at X,Y points, with '--' for the line)
plt.plot(x,y,color='g+--')
# explicit reference
plt.plot(x,y,color='green',marker='+',linestyle='dashed')

# Plot multiple items in one line chart
day=[1,2,3]
thing1=[50,36,45]
thing2=[45,25,55]
thing3=[51,26,36]
# labels:
plt.xlabel('Day')
plt.ylabel('Stuff')
plt.title('Random_chart')
# plot command:
plt.plot(day,thing1,label='t1')
plt.plot(day,thing2,label='t2')
plt.plot(day,thing3,label='t3')
# legend commands comes after the plot command:
plt.legend(loc='best')  # 'best' = best fit
# add grid in background:
plt.grid()





########
# Line graph showing iterations of cost function in gradient descent:
# graph representation of reduction of cost by # of iterations:
import matplotlib.pyplot as plt
cost = list(cost)
n_iterations = [x for x in range(1,50001)]
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(n_iterations, cost)



########################################################################
# Bar chart: (Requires Numpy to build Arrays)
########################################################################
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

company=['google','apple','tesla']
revenue=[50,36,45]
profit=[25,20,30]

# define your array (number of items in 'company'):
xpos = np.arange(len(company)) # this will build an array like: array([0, 1, 2])
# replace the array values with corresponding values in 'company'
plt.xticks(xpos,company)
# labels:
plt.ylabel("revenue (000)")
plt.xlabel("list of companies")
plt.title("graph title")
# plot
plt.bar(xpos,revenue, label="Revenue")
plt.legend()

# plot multiple items (revenue versus profit) -- Bar gets split
plt.bar(xpos,revenue, label="Revenue")
plt.bar(xpos,profit, label="Profit")
# Offset revenue/profit bars for distinction (its an array that why this works)
plt.bar(xpos-0.2,revenue, label="Revenue")
plt.bar(xpos+0.2,profit, label="Profit")
# bring bars side by side rev/profit by controlling width of bars:
plt.bar(xpos-0.2,revenue, width=0.4, label="Revenue")
plt.bar(xpos+0.2,profit, width=0.4, label="Profit")

# Horizontal bar chart (barh):
plt.barh(xpos+0.2,profit, width=0.4, label="Profit")



########################################################################
# Pie Charts:
########################################################################
expenses=[1400,600,300,410,250]
expenses_labels=["rent","food","phone","car","util"]

plt.axis("equal")
plt.pie(expenses,labels=expenses_labels)
plt.show() # this removes the system text
# Size/Radius of chart:
plt.pie(expenses,labels=expenses_labels, radius=2)
# Show percentages (with 2 decimal places:)
plt.pie(expenses,labels=expenses_labels, radius=1.5, autopct='%0.2f%%')
plt.show()

# Seperate pie pieces:
# create array map (equal to number of pieces)
# set them to zero
# then increase the value of the piece you want to seperate out of the pie:
plt.pie(expenses,labels=expenses_labels, explode=[0, 0, 0.5, 0, 0])
plt.show()

# Rotate pie graph:
plt.pie(expenses,labels=expenses_labels, startangle=180)
plt.show()



########################################################################
# Histograms
########################################################################

# plot histograms of all the dimensions in the table at the same time )1 graph for each)
# If the table has too many dimensions its gonna print a lot of graphs!!
%matplotlib inline
import matplotlib.pyplot as plt
TRAIN_df.hist()

# Pick a specific column to plot:
%matplotlib inline
import matplotlib.pyplot as plt
TRAIN_df["area"].hist()

# Define 'bins' or counting groups
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


# Details example of Hist()
import matplotlib.pyplot as plt
%matplotlib inline

# Only need a single dimension array for Histograms:
#(each value is from a different 'person')
blood_sugar=[113,36,45,50,67,82, 50, 49, 77, 90 ]
# by default it plots 10 'bucket/patches'/Bars
plt.hist(blood_sugar)

# custom range 'buckets'/Bars, and adjust width
plt.hist(blood_sugar, bins=3, rwidth=0.80)
# define range-groups explicity
plt.hist(blood_sugar, bins=[30,50,70,90,110], rwidth=0.80, color='g')

# multiple data sets, using arrays:
men=[113,36,45,50,67,82, 50, 49, 77, 90 ]
women=[110,26,65,30,77,52, 90, 49, 67, 100 ]

plt.xlabel('sugar range')
plt.ylabel('# of patients')
plt.hist([men, women], bins=[30,50,70,90,110], rwidth=0.80, color=['green','red'], label=['men','women'])
plt.legend()

# orient horizontally
plt.hist([men, women], orientation='horizontal')






########################################################################
# SCATTER PLOTS
########################################################################

# plotting from dataframe "Train_df"
# Call include .pyplot in import so you dont have to type plt.pyplot.

import matplotlib.pyplot as plt
%matplotlib inline
plt.xlabel='area'
plt.ylabel='price'
plt.scatter(TRAIN_df.area,TRAIN_df.price, color='red', marker='+')
### OR:
plt.scatter(df['area'],df['price'])

## Alternative syntax:
TRAIN_df.plot(kind="scatter", x="area", y="price")

# Reducing alpha to 0.1 highlights the high density points (number of incidents at long/lat coordinates)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

####### Advanced scatter plot (geographical housing prices, color coded)
# s = radius of circle (based on 'population' variable)
# c = price (cmap/jet option preset for blue=low, red=high)
housing.plot(
    kind="scatter"
    , x="longitude"
    , y="latitude"
    , alpha=0.4
    , s=housing["population"]/100
    , label="population"
    , figsize=(15,10)  # size of graph
    , c="median_house_value"
    , cmap=plt.get_cmap("jet")
    , colorbar=True
)
plt.legend()


####### plot from two dataframes on the same scatterplot (different color markers)
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='+')


########################################################################
# SCATTER PLOT + Line graph (for linear regression etc.)
########################################################################

import matplotlib.pyplot as plt
%matplotlib inline
plt.xlabel='area'
plt.ylabel='price'
plt.scatter(TRAIN_df.area,TRAIN_df.price, color='red', marker='+')
# plot the line, using the lin_reg result as the y-variable:
plt.plot(TRAIN_df.area, reg.predict(TRAIN_df[['area']]), color='blue')


###### Scatter + Line plot to show iterations (for Gradient Descent etc.)
% matplotlib
inline
def gradient_descent(x, y):
    m_curr = b_curr = 0
    rate = 0.01
    n = float(len(x))
    plt.scatter(x, y, color='red', marker='+', linewidth='5')
    for i in range(1000):  # adjust this to 10000 (if CPU can handle this)
        y_predicted = m_curr * x + b_curr
        plt.plot(x, y_predicted, color='green')
        md = -(2 / n) * sum(x * (y - y_predicted))
        yd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - rate * md
        b_curr = b_curr - rate * yd

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x,y)



########################################################################
# SCATTER MATRIX
########################################################################

# all combinations of the selected attributes will be plotted (so limit # of columns!)
from pandas.plotting import scatter_matrix
attributes = [
    "median_house_value"
    , "median_income"
    , "total_rooms"
    ,"housing_median_age"
]
scatter_matrix(housing[attributes], figsize=(12, 8))









########################################################################
# 3D Graph
########################################################################

########################################################################
# 3D plot
########################################################################

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# import train set as an array
# use 3 total variables and remove headers
# Order: x1, x2, y
# Area (independent), Bedrooms (independent), Homeprice (dependent)
test_array=np.loadtxt(open("homeprices_multi.csv"), delimiter=",", skiprows=1)
test_array


# split independent and dependent variables
x_train = test_array[:,[0,1]] #feature set
y_train = test_array[:,2] #label set

# PLOT: Actual Target Variable Visualization:
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
sequence_containing_x_vals = list(X_train.transpose()[0])
sequence_containing_y_vals = list(X_train.transpose()[1])
sequence_containing_z_vals = list(y_train)
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals,
           sequence_containing_z_vals)
ax.set_xlabel('Living Room Area', fontsize=10)
ax.set_ylabel('Number of Bed Rooms', fontsize=10)
ax.set_zlabel('Actual Housing Price', fontsize=10)