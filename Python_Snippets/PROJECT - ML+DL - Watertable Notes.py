

'''

Findings from DrivenData's Watertable Competition (Classification Problem with 3 targets)

1) Preprocessing Findings/Notes
2) Machine Learning Findings/Notes
3) Neural Networks Findings/Notes



##############################################################
# 1) Preprocessing - Findings/Notes
##############################################################

* Convert everything to numeric
* Handle numeric categorical values
------->> i.e. convert to dummy variables
--- Be prepared to revise preprocessing

Categorical Variables:
--- Avoid trying to create your own numeric ranking system (e.g. using averages to rank water sources)
!!!!!! NOTE !!!!! Make sure the same TRAIN average is used to fill the TEST set's NULLs !!!!!!!!!
--- subtract any redundunt columns created at this stage and counter dummy variable trap
--> Only try the manual ranking thing if there are too many dummy variables:
Too many dummy variables?
-- Start with all categorical variables which look reasonable, and then reduce if your CPU cant handle it


Null Handling:
-- If the distribution is a bell curve then using averages is totally fine
--> Otherwise.. maybe think of a different tactic (but averages are still reasonable)


Normalization:
I just divided by the max values of population etc. (setting all values between 0 and 1)
NOTE: make sure to use the same value for normalizing TEST set (use max from TRAIN set!)


---- Dependent Variables  y_train dataframe table (non array):
I used different formats for ML versus DL set
(research/ask which is best, I'm not sure..)
For ML algorithms:
I put the all target categories in 1 numeric column (three values: 1,2,3)
for DL Neural network:
I made 3 dummy variables (without subtracting any for trap)
>>> Problem here was that the result for DL is in probabilities for EACH option






##############################################################
# 2) Machine Learning Findings/Notes
##############################################################

- Create all the dummy variables in python, using the dummy variable function
- take out the redundant columns (make sure to avoid the dummy var trap)
- (my y_train had 3 numeric values)

-- The training tactics i used:
1) Compile a list of eligible candidates for models (linear vs classification etc.)
2) First run each model through cross_val_score individually (default parameters)
3) Note all default scores and shortlist your candidates for next step
4) Run the shortlist through GridSearchCV with only the most important parameters
5) Once the final model is selected do more fine tuning using cross_val_score or gridsearch (on only the final model)
6) Run a kfold validation to confirm the score\
7) I reran the grisearch with the final parameters (step5) to train the final version of the model
(at this point you wanna test the model on the TEST set and save the model in case this isnt an adhoc)
8) Proceed with the final trained model to predict the result of x

Result: highest score was 78% (submitted accuracy) with Random Forests
>>> this was 2% better than the score i got with my neural net
>>> Top score (so far) on the leaderboard is 83%

##############################################################
# 3) Neural Networks Findings/Notes
##############################################################

Setup 1)
y_train had 3 binary target columns (each column for 1 possible option)
-- Final output layer: Activation = 'sigmoid' (for binary), and 3 neurons (one per option)
-- For Input and Hidden layers activation = 'relu'
-- Metrics: 'accuracy' (for percentage)
-- Loss: binary_crossentropy
-- Call the history() function for callbacks in fitting step (to plot loss function graph)
NOTE: if you have both Training and Testing sets
>>>> model.fit(X_train, y_train, validation_data=(X_test,y_test), callbacks=[history]))
If you dont have a train/test split:
model.fit(X_train9, y_train2, epochs=10, validation_split=0.33, callbacks=[history])

Setup 2)
tried y_train with 1 column, with 3 values
This required:
Output Layer: model.add(Dense(3,activation = "softmax"))
loss Function "sparse_categorical_crossentropy"
The result was automatically converted into three output probability columns (when converting to dataframe)
But the accuracy was worse, with all else being equal... so stuck with Setup 1
----->> Best to use y_train with dummy variable targets (without subtracting one for dummy trap)


-- Tuning Notes:
1) Number of layers (start with 1 hidden (3 total))
2) Number of Neurons in each layer  (start small, like 25/30/35)
3) Learning Rate (start with 0.0001 and go up)
>> Check graph for loss rate, it should look like decreasing 'elbow' graph
4) Number of Epochs (start with 3 and go up... high as possible)
Trail and Error... play around with it... check the loss and accuracy plots
5) End by confirming accuracy with the KFold validation script for NNs (submitted accuracy was the same though)


Prediction Table Formatting:
The output dataframe will have probabilities in each target column
Pick the highest value - I exported back to SQL and used a CASE function


RESULT: Best score was 76% (2% less than random forest)



'''



