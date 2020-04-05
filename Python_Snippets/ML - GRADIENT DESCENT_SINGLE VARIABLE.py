


########################################################################
# GRADIENT DESCENT
########################################################################
'''
Here, you have to manually find the 'best fit line' iteratively
>> you still start with a 'train' set with X and Y provided
>> But now you have to derive the y=mx+b equation, by finding the best 'm' and 'b'

The goal (like in Linear Regression) is to find the best slope('m') and y-intercept('b') for y=mx+b
The 'Mean Square Error (MSE)'/'Cost Function' is used to do this.

This is an iterative method: ('cost/MSE', 'm' and 'b'), which depend on the X and Y you feed
The goal is to reach the minimum value for the 'Cost/MSE' value, with respect to 'm' and 'b'
The value of 'm' and 'b' at this lowest value of 'cost/MSE' is selected as the 'best fit' for y=mx+b

(This is a Multivariable Calculus problem, with partial derivitives + chain rule)
'''


############################################################################
########## Pycharm code, will print 10000 iterations for debugging/optimizing
import sys
sys.path.append("X:\Anaconda\Lib\site-packages")
import numpy as np

# Find the best fit line for 'm' and 'b'
# Define parameters (before loop) and adjust them to fine-tune the algorithm
def gradient_descent(x,y):
    m_curr = b_curr = 0  # starting value for m and b
    iterations = 10000     # number of steps to find the global minimum (start with 1K and then refine)
    n = float(len(x))         # !! Make sure X and Y training sets are of equal rows (without NULLs/missing data) !!
    learning_rate = 0.08  #starting parameter (start with 0.001 and then refine)

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])  # cost/MSE function
        md = -(2/n)*sum(x*(y-y_predicted))  # derivitive of m
        bd = -(2/n)*sum(y-y_predicted)  # derivitive of b
        m_curr = m_curr - learning_rate * md  # (m = m - learning rate * d/dx(m))
        b_curr = b_curr - learning_rate * bd  # (b = b - learning rate * d/dx(b))
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))  # shows results for each iteration

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

# now check the iterations, the cost should be going down to Zero
gradient_descent(x,y)


############################################################################
# Jupyter graph code: !!! NOTE: 10K iterations takes time to generate the graph !!

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
def gradient_descent(x,y):
    m_curr = b_curr = 0
    rate = 0.01
    n = float(len(x))
    plt.scatter(x,y,color='red',marker='+',linewidth='5')
    for i in range(10000):        # !!! adjust iterations here, if CPU is overloaded !!!
        y_predicted = m_curr * x + b_curr
        plt.plot(x,y_predicted,color='green')
        md = -(2/n)*sum(x*(y-y_predicted))
        yd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - rate * md
        b_curr = b_curr - rate * yd

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)

# Final step: Output
# Now that you have your 'm' and 'b', use it on your TEST dataset
# x_test  (create sample, or import)
x_test = {
	'x' : [1,2,3,4,5,6,7,8,9,10],
}
x_test = pd.DataFrame(x_test)

# define function of y=mx+b
# m = 2
# b = 3
# x is your input
# y is your result
def res_func(x):
	y = 2*x +3
	return y

# create output dataframe:
result_df = res_func(x_test)
# bring in result to the original test df as a 'result' column
x_test['result'] = result_df
# output file
x_test.to_csv("prediction_GradientDescent.csv", index=False)