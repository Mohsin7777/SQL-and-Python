
# PyCharm Import:

# (from main directory:)
# import sys
# sys.path.append("X:\Anaconda\Lib\")
# import urllib as urllib

# (from site packages:)
# For pandas:
# import sys
# sys.path.append("X:\Anaconda\Lib\site-packages")
# import pandas as pd



################################################################
# examples below have print() for everything (for pycharm testing)
# but in jupyter, you only need print() for specific cases, like stuff inside loops
################################################################



################################################################
# Variables
################################################################

rent=1220
gas=202.5
groceries=605.6
# NOTE: Can't use reserve words or special characters

# PRINT a variable
print(rent)

# Calculate Variables 
total=rent+gas+groceries
print(total)

# Update Variable (will require updating/refreshing the connected variable)
rent=1440
total=rent+gas+groceries
print(total)

# Store the name of variables
item1="rent"
item2="gas"
item3="groceries"
print("Expenses:",item1,item2,item3)




################################################################
# Numbers
################################################################

#Basic aritmetic operations
print(2+2)
print(2-2)
print(2/2)
print(2*2)

#Find Remainder: divide 11 by 2 and print remainder
print(11%2)

# Powers: 3^2
print(3**2)

#Combine with Variables (distance between cities)
nyc_bal=188
nyc_pitt=247
total_distance=nyc_bal+nyc_pitt
print(total_distance)
mph=65
time=total_distance/mph
print(time)

#round FLOAT to integer or define precision
print(round(time))
print(round(time,2))

# !! NOTE: programming languages don't store floating numbers well
print(6-5.7) #incorrect output - should be 0.3
print(round(6-5.7,2))  #correct output required rounding to 2 decimal places, to get 0.3

# return true/false value
3>2 and 1<2

# Comma thousands seperator
'{:,}'.format(1000)

################################################################
# STRINGS
################################################################

#Python stores text in a sequence of characters in memory
# single or double quotes dont matter in this case
text = 'ice cream'
print(text)

# Access characters in memory using index (cannot modify, unlike items in lists)
print(text[0])
print(text[1])
print(text[2])

#SUBSTRING
print(text[0:3]) 
print(text[4:]) #prints all after 4th character

# Use double quotes to handle single quotes inside string
text="let's learn python"
print(text)

# Use single quotes to handle double quotes inside string
text='say "Hello World"'
print(text)

# Triple Quotes for Multi-line String
# (python stores this using '\n' to denote line break)
address = '''1 purple street
new york
usa'''
print (address)

# CONCATENATE
s1="hello"
s2='world'
print(s1+' '+s2)

# CONCATENATE STRING WITH NUMBER
num=25
print('number='+str(num))



################################################################
# LISTS
################################################################

# Create list
items=["bread","pasta","veggies","chicken"]
print(items)

# Index for list
print(items[0])
print(items[1])
print(items[3])

# Modify list  (unlike characters within a string, items in a list can be modified with index operations)
items[0]='chips'
print(items)

# Print first two items 
# Note: cant say 0:1 because second index is excluded
print(items[0:2])

# Last item, can use negative index
print(items[-1])

# Append to list
items.append("butter")
print(items)

# Insert Item in list at specific location using index
items.insert(2,'coke')
print(items)

# Merge Lists
# > Combined_list = List1 + List2

# LEN function on list returns number of ITEMS within list
print(len(items))

# IN operator checks for items in a list
print("chips" in items)
print("soda" in items)




################################################################
# IF statements ( CASE )
################################################################

''' Simple IF/ELSE:

#Print text which expects an input
num=input("Enter a number: ")
#define datatype of input (it is string by default, so convert it to integer)
num=int(num)

# IF statement: (divide by -1 and figure out the remainder)
if num%2==0:
	print("number is even")
else:
	print("number is odd")
'''

''' IF / ELIF (Else, run another IF)

indian=["samosa","daal"]
chinese=["rice", "chicken"]
italian=["pizza","pasta"]

dish=input("Enter a dish name:")

if dish in indian:
    print("indian")
elif dish in chinese:
    print("chinese")
elif dish in italian:
    print("italian")
else:
    print("the hell is this:", dish)

'''


################################################################
# FOR LOOP
################################################################

''' 

# Simple FOR loop that adds values in a list

expenses = [2340, 2500, 2100, 3100, 2980]

total = 0
for item in expenses:
    total = total+item
print(total)
'''


'''

# Print 1 to 10 using range()

for i in range(1,11):
    print(i)
    
# prints square of numbers in range

for i in range(1,11):
    print(i*i)

# doubling function ( exponential? )    
for i in range(1,11):
    print('iteration #', i, ': ', 2**i) 
    
# cleaner output
for i in range(10,51):
    print('iteration #', i-10, ': ', '{:,}'.format(2**i))
'''

'''
# Print string value with each result

expenses = [2340, 2500, 2100, 3100, 2980]
total=0
for i in range(len(expenses)):  # this means iterate 5 times, for the 5 items in the list
    print('Month:',(i+1),'Expense:',expenses[i])
    total = total + expenses[i]

print('total expense is:', total)
    
'''

'''
# Use BREAK to break loop when condition is met:

key_location = "chair"
locations=["garage","living room", "chair", "closet"]
for i in locations:
    if i==key_location:
        print("found in ",i)
        break        # IF above criterea is met, it will break the loop here
    else:
        print("not found in ",i)
'''

'''
# Use CONTINUE to pass over items
### e.g. print square of only EVEN numbers between 1 and 5:

for i in range(1,6):
    if i%2==0: # checks for remainder after dividing by 2
        continue  # IF above criterea is met, it will SKIP next step and CONTINUE from beginning
    print(i*i)
    
'''


################################################################
# WHILE LOOP
################################################################

'''
# WHILE loop:

i=1
while i <= 5 :
    print(i)
    i = i + 1   # make sure to increment this counter, otherwise it will run forever
    
'''



################################################################
# FUNCTIONS
################################################################


#### Simple Function:
total=0  # this is a global variable is outside the function

def sum(a, b):
    print("a:",a)
    print("b:",b)
    total = a + b     #this is a local variable (not the same as outside 'total')
    print("total inside function:",total)
    return total

n = sum(b=5, a=6)  #you can reorder items if you specify. The default order is defined in function definition

print('total outside function:', total)  #the global variable outside is still 0


# USE DEFAULT VALUES TO ALLOW FLEXIBILITY IN INPIT
def sum(a=0, b=0): #Enter default values
    total = a + b
    return total

p=sum(a=1) #Now you dont need to put in b variable
print(p)



'''

#### Function: Use the same code for different variables (input/output)

# Original code requires 2 seperate loops:
tom_exp_list = [2100,3400,3500]
joe_exp_list = [200,500,700]

# One for Tom
total=0
for item in tom_exp_list:
    total = total + item
print("Tom's total:",total)

# Second for Joe
total=0
for item in joe_exp_list:
    total = total + item
print("Joe's total:",total)


########## Now use Function instead:
def calculate_total(exp):       #'exp' is a local variable (argument) which will be passed
    total = 0                   #function body
    for item in exp:            #function body
        total = total + item    #function body
    return total                # this is the return value

# Now Call the variable in a function, instead of writing 2 loops:
tom_total = calculate_total(tom_exp_list)
joe_total = calculate_total(joe_exp_list)

print("Tom's total:",tom_total)
print("Joe's total:",joe_total)

'''



################################################################
# DICTIONARIES
################################################################

#Dictionary items can be retrieved without using index
# Order of items doesnt matter in dictionaries

# Dictionary of phone numbers
d={"tom":55555555, "rob":66666666}

# return value in dictionary
print(d["tom"])

# add value in existing dictionary
d["sam:"]=77777777
print(d)

# Delete item from dictionary
del d["tom"]
print(d)

# Print all the directory values using a FOR loop

### Solution 1)
for key in d:
    print("name:",key,"value:",d[key])  #call 'key' whatever you want

### Solution 2)
for key,value in d.items():  #use inbuilt function items()
    print("name:", key, "value:",value)


# Check if item is in list
print("rob" in d)
print("bob" in d)

# Clear all items within dictionaries
d.clear()
print(d)


################################################################
# TUPLES (versus Lists)
################################################################

## Tuples are lists of DIFFERENT types of values grouped together
## In a List, all values have the same meaning (Homogenous)
## In a Tuple, all values have a different meaning (Heterogenous)
## Note: Unlike Lists, Tuples are also immutable (can't change items)

# e.g. Represent a geometric point in 2D, where x,y values represent different/'Heterogenous' concepts
point=(5,9)  #note () when defining, where lists are defined using []
print(point[0]) #note square brackets when calling tuples, just like lists
print(point[1])

# If all items were X values, it would be a LIST e.g. X=[4,6,5,2]
# note square brackets for defining lists [], but when calling items, both tuples and lists use []






################################################################
# JSON files
################################################################

# JSON data is in a STRING, but can be converted
# Create JSON formatted text
book = {}
book['tom'] = {
    'name': 'tom',
    'address': '1 red street, NY'
}
book['bob'] = {
    'name': 'bob',
    'address': '1 green street, NY'
}

import json
s=json.dumps(book)
print(s)

# write to a file:
s=json.dumps(book)
with open('X://Python_Scripts/book_JSON.txt',"w") as f:
    f.write(s)

# read JSON records
f=open('X://Python_Scripts/book_JSON.txt',"r")
s=f.read()
print(s)

# Parse JSON string and convert it into Dictionary
json.loads(s)  #'load string'
book=json.loads(s)
print(book) #this is now a dictionary (not a string)
print(type(book)) #confirm that it is a dictionary

# Print items within JSON dictionary
print(book['bob']['address'])

# Print all items, using FOR loop
for person in book:
    print(book[person])


################################################################
# ENTRY POINT:
# if __name__ == "__main__"
################################################################

# Designed to be used at the start of the script/module
# to make sure the script is running in the right module etc.

# preset Python variable "__name__"  is set to  "__main__"
print(__name__)
# This is similar to C/Java, it is called an "ENTRY POINT"

# If this script is run from within the main script/program/module, its value will stay as "__main__"
if __name__ == "__main__":
    a=print("hello world")
    print(a)

# But if this script was imported as a module into another script, the value of __name__ will change


################################################################
# Exception Handling
################################################################

# Handle Any Generic Error, with alternative action
try:
    print(1/0)
except Exception as e:
    print("Exception occured!")

# Handle specific exception (division by error), with alternative action
try:
    print(1/0)
except ZeroDivisionError as e:
    print("Exception occured!")


'''
# Handle multiple exceptions, with alternative action
x=input("enter num1: ")
y=input("enter num2: ")

try:
    z = int(x) / int(y)
except ZeroDivisionError as e:
    print("Divide by Zero Error!")
    z = None
except TypeError as e:
    print('Type Error!')
    z= None
except ValueError as e:
    print('Value Error!')
    z= None
print("Division is: ", z)
'''

################################################################
# Class and Objects
################################################################

# For HUMANS (Class) - It is an abstraction of some entity
## Contains Properties: e.g. Name, Gender, Occupation
## Contains Methods: Speaks, Works, Sleeps

# Objects are specific examples of a class
## Tom Cruise is an object, with properties and methods
## Roger Federer is another object

#Create class "Human" with properties and methods:
class Human:
    def __init__(self, name, occupation):
        self.name = name
        self.occupation = occupation

    def do_work(self):
        if self.occupation == "tennis player":
            print (self.name, "plays tennis")
        elif self.occupation == "actor":
            print (self.name,"makes movies")

# Create object, "self" is auto passed, only pass the rest of the argument
tom = Human("tom cruise","actor")
federer = Human("federer","tennis player")

# Call objects:
tom.do_work()
federer.do_work()

