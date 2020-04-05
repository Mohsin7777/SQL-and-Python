


# examples below have print() for everything  (for pycharm testing)
# but in jupyter, you only need print() for specific cases, like stuff inside loops


##########
'''
NOTE - BEFORE IMPORTING INTO PYTHON, PREPROCESS DATASET IN SQL:

1) START BY REDUCING YOUR DATA TO 1 FINAL TABLE (ALL JOINING/LOGIC FINALIZED)
2) NULL HANDLING
3) SIMPLIFY DIMENSIONS (E.G. DONT NEED 100 CITIES WHEN PROVINCES WILL DO)
4) REMOVE OUTLIERS, ERRORS AND INSIGNIFICANT VALUES, IN ALL DIMENSIONS
5) IF POSSIBLE, CONVERT ALL VARCHAR TO NUMERIC
>> CREATE DUMMY VARIABLES FOR ALL CATEGORIES (WHILE AVOIDING 'DUMMY VARIABLE TRAP')
>> FOR MESSIER STRING DATA, YOU CAN USE PYTHON FUNCTIONS AFTER IMPORT
6) DROP ANY UNNECESSARY DIMENSIONS
'''
##########





################################################################
# Connect to SQL Server

'''NOTE: SQL Server reorders the rows based on first column
 Also watch out for the ordering if Train/Test splits are seperate tables'''
################################################################

import pandas as pd
# connection string (this connects with windows authentication, launch SQL Server with admin mode):
from sqlalchemy import create_engine
from sqlalchemy import create_engine
sqlcon = create_engine('mssql+pyodbc://@' + 'GHOST-117\SQLEXPRESS' + '/' + 'MOHSIN' + '?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server')
# SQL
df = pd.read_sql_query("SELECT DISTINCT [Pclass],[Sex],[Age],[Fare],[Survived] FROM  MOHSIN.DBO.TITANIC_PREPROCESSED", sqlcon)

# to bring in an entire table:
df = pd.read_sql_table("table_name", sqlcon)
# specific columns:
df = pd.read_sql_table("table_name", sqlcon, columns['c2','c3'])

# Write table 'df2' into SQL Server
df2.to_sql(
    name='test', #the name of the created table
    con=sqlcon,
    index=False,
    schema='MOHSIN.dbo',
    if_exists='fail'  #fail, replace, append
)



####################################################################
####################################################################
# Jupyter Notes:
####################################################################
####################################################################

# Install Anaconda package using downloadable installer
# Jupyter notebook is a web app that allows live code & embedded links and visualizations
# launch jupyter lab from script directory:
# > cd /D X:\Python_Scripts\Jupyter_Stuff
# launch command:
# > jupyter lab
# > import pandas as pd

# note: pandas.io.data has been deprecated
## replaced with: pandas_datareader

# Anaconda package
# this comes with conda package manager, which allows installation of non python library dependencies


####################################################################
# Pandas in Jupyter:
####################################################################

####################################################################
# CREATING DATAFRAMES i.e. TABLES

#Documentation APIs: "Pandas IO" / IO Tools
####################################################################


# 1) IMPORT FILES (panda)
# CSV 
df = pd.read_csv("weather_data.csv")
# EXCEL, and can define individual sheets
df = pd.read_excel("weather_data.xlsx","Sheet1")
# TXT (numpy):
df = np.loadtxt('data2.txt', delimiter=',')

#2) CREATE using TUPLES
# this is just a collection of rows, and provide column names at the END
weather_data = [
    ('1/1/2017',32,6,'Rain'),
    ('1/2/2017',35,7,'Sunny'),
    ('1/3/2017',28,2,'Snow')
]
df = pd.DataFrame(data=weather_data, columns=['day','temperature','windspeed','event'])
df

# 3) CREATE using DICTIONARY
# Each 'key' contains the row values
# (requirement: the number of rows in each column have to be equal(not sure how to add NULL))
test_table = {
	'column1' : [1,2,3],
	'column2' : ['1/1/2020', '1/2/2020','1/3/2020'],
	'column3' : ['a','b','c']
}

df = pd.DataFrame(test_table)

#4) CREATE using DICTIONARY LIST 
# Difference is that each element represents 1 row, with column header provided
weather_data = [
    {'day': '1/1/2017', 'temperature': 32, 'windspeed': 6, 'event': 'Rain'},
    {'day': '1/2/2017', 'temperature': 35, 'windspeed': 7, 'event': 'Sunny'},
    {'day': '1/3/2017', 'temperature': 28, 'windspeed': 2, 'event': 'Snow'},
    
]
df = pd.DataFrame(weather_data)
df




####################################################################
# BASIC PRINT COMMANDS
####################################################################

# Check dimensions of table:
# first do this:
rows, columns = df.shape
# then run either command: rows/columns
rows
columns

#Print all the column headers
df.columns

#Print first N rows (enter # within brackets)
df.head()



####################################################################
# SELECT statement
####################################################################

#TO SELECT ENTIRE TABLE (after importing/creating df)
df  

#SELECT column
df['column_name']

#SELECT multiple columns (needs extra [])
df[['column_1','column_2']]

#check data type
type(df.column_name)

#max / min value
df['column_name'].max()

#standard deviation
df['column_name'].std()

# show a bunch of quick statistics (over all the numeric dimensions)
df.describe()



####################################################################
# INDEX
####################################################################

# shows current index
df.index

# set index (!!make sure this is unique!! (except for special cases))
df.set_index('column_name', inplace=True)

# use index to print info, pivoted on the index value
df.loc['index_value']

# Reset/Remove index
df.reset_index(inplace=True)

####################################################################
# DATETIME stuff
####################################################################

# basic date parse (string to date):
df = pd.read_csv("file.csv", parse_dates["column_name"])

# Use the DATE column as INDEX (first convert the value to date with parse)
df = pd.read_csv("file.csv", parse_dates["column_name"], index_col["column_name"])
# now you can use partial index to retrieve all of the month etc.
df["2020-01"]
# can also add metrics:
df["2020-01"].mean()
# can use range of dates:
df['2020-01-01':'2020-01-20']

# Define Datetime periods:
m = pd.Period('2020-1',freq='M')
# print out the starttime/endtime
m.start_time / m.end_time
# operations:
m+1 (adds febuary)


# TIMEZONES:
from pytz import all_timezones
print(all_timezones) #this will show all the timezones available
# python has 2 datetime objects: 'Naive' (no timezone) and 'Timezone Aware'
# assign timezone to date index:
df = df.tz_localize(tz='US/Eastern')
df.index # now it will show timezone with UTC-4
# Convert to Berlin:
df = df.tz_convert(tz='Europe/Berlin')
# Combine datasets for different timezones (b dataset) and (a dataset)
print(a+b) #will synchronize to UTC


# HOLIDAYS:
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
# use
holiday = CustomBusinessDay((USFederalHolidayCalendar())
# Now you can use 'holiday' in the frequency type arguments to exclude/select holidays etc.
# the following excludes 4th of july holiday
pd.date_range=(start='7/1/2020',end='7/21/2020',freq=holiday)


#######################
# resample()
# Object must have datetime like index
#######################

# Monthly Average of "sales" in df:
df.sales.resample('M').mean()
# Weekly
df.sales.resample('W').mean()
# For all the frequencies check documentation (e.g. D=day, Q=quarter-end, A=year-end, H=hourly etc.)

# Plot Monthly chart:
%matplotlib inline
df.sales.resample('M').mean().plot(kind="bar")



####################################################################
# Read/Write files
# NULL handling
####################################################################

# Ideally, move relevant files to the Jupyter directory and launch Lab from that location
# For subfolders within the root directory: df = pd.read_csv("FOLDER/FILE.csv")

# READING CSV:
df = pd.read_csv("4_read_write_to_excel/stock_data.csv")
# Removing headers:
df = pd.read_csv("4_read_write_to_excel/stock_data.csv", skiprows=1)
# Adding headers:
df = pd.read_csv("4_read_write_to_excel/stock_data.csv", header=None, name=["ticker","eps"])
# Limited rows - (excludes headder)
df = pd.read_csv("4_read_write_to_excel/stock_data.csv", nrows=3)

# Convert words to numbers e.g. One = 1 , with word2number module
from word2number import w2n
print(w2n.word_to_num("one"))

###############
# NULL HANDLING:
###############

# Replace NULL with 0 in the entire dataframe
new_df = df.fillna(0)

## fillna() conditional, based on column:
new_df = df.fillna({
    'column1': 0,
    'coumn2': 'zero'
})

# forward fill (take previous row value and fill in the blanks (vertically)
df.fillna(method="ffill")
# backward fill:
df.fillna(method="bfill")
# to fill horizontally:
df.fillna(method="ffill", axis="columns")
# Use ffill with limit (limit how many times to copy forward/backward)
df.fillna(method="ffill", limit=1)

# Fill null with an average over the column ("math.floor" rounds down: e.g. 3.5 to 3):
import math
avg_of_column = math.floor(df.column_name.median())
df.new_df = df.new_df.fillna(avg_of_column)


# Fill NULL with with an estimate, based on previous value:
df.interpolate() #default is linear estimate
# see documentation for options: Date, index, nearest, time, cubic etc.

# delete rows with any NULL values:
df.dropna()
# delete rows with X number of NULLs e.g. all values
df.dropna(how="all")
# threshold for non-NULL values (keep row if at least X number of values exist)
df.dropna(thresh=1)

# Insert Missing Dates for NULL and Recreate the index
dt = pd.date_range("01-01-2020","01-11-2020") # Create date range:
idx = pd.DatetimeIndex(dt) # pass that range to an index
df = df.reindex(idx)  # reindex your dataframe


####################################################################
# File Cleaning
# REPLACE values  - Strings, REGEX cleaning etc.
####################################################################

# with Pandas (when importing):
# Change random values to NULL:
df = pd.read_csv("folder/file.csv", na_values=["random string","another random string"])
# Conditionally change random values to NULL, column specific:
df = pd.read_csv("folder/file.csv", na_values={
    'column1': ["random1","random2",-1],
    'column2': ["random1",-10023],
})

# with numpy
# Replace values in entire dataframe:
new_df = df.replace('whatever value',np.NaN)
### replace multiple values:
new_df = df.replace(['whatever value',00000],np.NaN)
# for specific columns:
new_df = df.replace({
    'column1': 'random value to replace'
    'column2': -99999
},np.NaN)

# replace something with anything:
new_df = df.replace({
    'random value1': np.NaN # this replaces it with NULL
    'random value2': 0      # this replaces it with a zero
})

########
#REGEX
# clean strings with REGEX
# See documentation for all commands, below is a common example

# replace all alphabet characters with blanks:
new_df=df.replace('[A-Za-z]','',regex=True)
## column specific
new_df=df.replace({
    'column1': '[A-Za-z]',
    'column1': '[A-Za-z]'
},'',regex=True)
########


#####################################
# CASE-like function: (Lambda)

# Convert columns to number using CASE-like function:
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)



####################################################################
# Replace using REFERENCE TABLES (Lists & Dictionaries)
####################################################################

# This Replaces string values with numbers, using your List/Reference
new_df=df.replace(['value1', 'value2'], [1,2])
# converts 'value1' to 1, 'value2' to 2 .... etc.


# This maps values from a reference set, using built-in Lambda function in pandas
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
# convert into dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# reference for the target names
iris.target_names
# LAMBDA FUNCTION:
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

###############
# WRITING TO CSV
###############
df.to_csv('file.csv')
# removing default index:
df.to_csv('file.csv', index=False)
# write specific columns
df.to_csv('file.csv',columns=['column1','column2'])
# skip writing header
df.to_csv('file.csv', header=False)

###############
# EXCEL
###############

#basic:
df = pd.read_excel("weather_data.xlsx","Sheet1")

# Reading with Cell Convertor (conditional function) when importing (column specific)
def name_of_function(cell):
    if cell=="n.a.":
        return 'value you want'
    return cell

df = pd.read_excel("folder/file.xlsx","Sheet1", converters = {
    'column': name_of_function
})

# Writing excel: (To offset:) startrow, startcol, (can also use same CSV arguments like index=Fale etc.)
pd.to_excel("folder/file.xlsx",sheet_name="sheet1")

# Writing multiple dataframes to seperate sheets:
with pd.ExcelWriter('file.xlsx') as writer:
    df_first.to_excel(writer, sheet_name="sheet1")
    df_second.to_excel(writer, sheet_name="sheet2")







####################################################################
# WHERE CLAUSE Filters
# Documentation: "Pandas Series operations"
####################################################################
df[ df['column_name'] == value]
#e.g.
df[ df['Temperature'] >= 32 ]
# combine with other functions like max/min:
df[ df['column_name'] == df['column_name'].max()]

# SELECT single column with WHERE clause
df['day'] [df['temperature']==df['temperature'].max()]
# ^ returns only the selected column (in this case, the date of max temp)

# select multiple columns (needs extra [] ):
df[['day','temperature']] [df['temperature']==df['temperature'].max()]



####################################################################
# GROUP BY
# Documentation: pandas groupby (split apply combine)
####################################################################

# Goal (like SQL) is to use Group to split the data (here it is literally split into an "object" with subtables)
## then apply some calculation
## and then recombine into a new dataset


# Create a 'group by object' for the column 'city'
g = df.groupby('city')
# this creates a new object where the dataframe is grouped by 'city'
# this object is a collection of dataframes, one for each city value in the 'city' column

# print each grouped sub-table, using a for loop:
for city, city_df in g:
    print(city)
    print(city_df)

# get specific subset
g.get_group("toronto") #enter a value of the city

# Apply some calculation on grouped object:
g.max() # use g.describe() for all basic metrics
# This will output max values for each city, recombined into one dataset

# plotting with group by:
%matplotlib inline
g.plot()




####################################################################
# Merge Dataframes  (i.e. JOIN)
# .... seriously, just use SQL for this stuff !
####################################################################

# INNER JOIN on 'City' column
df3=pd.merge(df1, df2, on="city")

# UNION/Outer Join
df3=pd.merge(df1, df2, on="city", how="outer")

# LEFT join
df3=pd.merge(df1, df2, on="city", how="left")



####################################################################
# Pivot
# excel like pivots, check documentation  (uses numpy functions for aggregates)
####################################################################

df.pivot_table(index="city",columns="date", aggfunc="sum")



####################################################################
# Concatenate Dataframes  (its like Append/Insert one table into another)
# Documentation: pandas concat
####################################################################

# select dataframe names of tables and create new default index:
pd.concat([table_1, table_2], ignore_index=True)

# pass keys (like labels), to differentiate subsets (will recreate a new index automatically)
pd.concat([table_1, table_2], keys=["key1","key2"])
# select specific subset
df.loc["key1"]





####################################################################
# Numpy - Array / Matrix operations / Linear Algebra stuff
####################################################################


''' #import in pycharm:
import sys
sys.path.append("X:\Anaconda\Lib\site-packages")
import numpy as np
'''

# Convert a dataframe to numpy array: (this will convert only the values in the dataframe, not the index)
uni_data = uni_data.values
# >>> see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html


# One dimensional array (looks like a list)
a = np.array([5,6,7])
print(a[0])

# Two dimensional array
a = np.array([[1,2],[3,4],[5,6]])
print(a)

# check how many dimensions an array has:
print(a.ndim)
# byte size for items
print(a.itemsize)
# print datatype
print(a.dtype)
# number of elements in array
print(a.size)
# Shape (length/width of array)
print(a.shape)

# specify datatype of array:
a = np.array([[1,2],[3,4],[5,6]], dtype=np.float64)

# initialize array with default zeros
a = np.zeros( (3,4) )
print(a)
# initialize array with default ones
a = np.ones( (3,4) )
print(a)


# initialize array using aRange function (this will have 1,2,3,4)
a = np.arange(1,5)
print(a)
# initialize array using aRange function, with STEPS of 2 (skips over 2 and 4)
a = np.arange(1,5,2)
print(a)
# initialize array using Linearly Spaced Values: specify 1) Start, 2) End, and 3) linear spacing
a = np.linspace(1,5,20)
print(a)

# Reshaping Arrays:
a=np.array([[1,2],[3,4],[5,6]])
print(a)
print(a.shape)
## Reshape it to be 2 by 3 (have to copy it to a new array)
b = a.reshape((2, 3))
print(b)
# flatten array
b = a.ravel()
print(b)

# max/min/sum for the entire array:
print(a.min())
print(a.max())
print(a.sum())

# max/min/sum for the rows/columns: Axis 0 = Rows, Axis 1 = Columns
print(a.sum(axis=0))
print(a.sum(axis=1))

# can also use other numpy functions e.g.
#square root
print(np.sqrt(a))
#standard deviation
print(np.std(a))


# Basic Linear Algebra operations on Matrices
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])

# Can Add/Substract/Multiply/Divide arrays:
print(a+b)
# Matrix Product / Dot Product
print(a.dot(b))

# NOTE: for Indexing/Iterating/Stacking arrays - check other tutorials







