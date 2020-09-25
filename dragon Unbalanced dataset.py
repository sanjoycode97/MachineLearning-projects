"""we have to split the dataset according to the unbalanced data column which has the unbalanced number of catagorical data
so we have to shuffled split the data
"""
import pandas as pd
import numpy as np
#load the data first
data=pd.read_csv("project_data.csv")
print(data.head())
#describe the dataframe
print(data.info())
#make the copy of the data frame because we will do every operqation  with copy of the dataframe
data_copy=data.copy()
print(data_copy.head())
#now check the categorical value col which is in the interger values with 1 and 0
print(data_copy["CHAS"].value_counts())# it will rfetrun the how mane categorical value exist int hat columns
#here 1 and 0 values are not in the same match ratio is very high
#so we have to take care of the data set while the split the dataset into the train test
# we need to shuffeled dataset and then split into the train and test set
from sklearn.model_selection import StratifiedShuffleSplit
strat=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
#now split the data set into the train dataset and test dataset
for train_index,test_index in strat.split(data_copy,data_copy["CHAS"]):#according to the chas col
    train_set=data_copy.iloc[train_index]
    test_set=data_copy.iloc[test_index]
#now check the values of chas column in the train and test dataset
print(train_set["CHAS"].value_counts())
print(test_set["CHAS"].value_counts())
#now we have split the data into the train test set
#check the train and test data set
print(train_set)
print('\n')
print(test_set)
#now split the variable into the train and test dependent and independent variable
#split the independent and dependent variable of train set
x_train=train_set.drop("MEDV",axis=1)
y_train=train_set["MEDV"]
#now split the test data set into the dependent and independent varibale
x_test=test_set.drop("MEDV",axis=1)
y_test=test_set["MEDV"]
# lets check the dependent and independent variable of both train and test dataset
print(x_train)
print(y_train)
print("\n")
print(x_test)
print(y_test)