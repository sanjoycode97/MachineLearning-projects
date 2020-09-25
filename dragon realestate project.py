"""
here we have to evaluate our model on Root mean squared error
"""
"""
Dragon Real Estate Price predictor
"""
import pandas as pd
housing=pd.read_csv("project_data.csv")
print(housing.head())
print(housing.info())
print(housing["AGE"].value_counts())
print(housing["CHAS"].value_counts())
"""
here we want to check our data frame data all the details of the every columns
MEAN,COUNT of every columns's values,Standard deviation,min,max
Standard deviation:-how data is spread out from the mean of the data
"""
print(housing.describe())
#visualizing the data
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()
#train test set learning purpose coding
"""
##Train test spliting function
import numpy as np
#to stabilize the random seed to prevent random number repeatedly
np.random.seed(42)
def split_train_split(data,test_ratio):
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
train_set,test_set=split_train_split(housing,0.2)
print("length of train set data: ",len(train_set))
print("length of test set data: ",len(test_set))

"""
# split the main data set into the train test data set using sklearn library
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"length of the train set: {len(train_set)} and length of the test set: {len(test_set)}")
"""
equal partition of train and test data for similar kaind of data can machine train and test on
"""
#here in the have CHAS column not similar no of categorical data so we have to split the data set by stratified sampling
#so we split the data by the tratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
strat=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in strat.split(housing,housing["CHAS"]):#here we have to provide from which column respect we have to split the data
    strat_train=housing.iloc[train_index]
    strat_test=housing.iloc[test_index]
#now copy the train dataframe
housing=strat_train.copy()
#Looking for correlation
corr_matrix=housing.corr()
print(corr_matrix["MEDV"].sort_values(ascending=False))#here we see correlation of feature variable with label housing["MEDV"]
#here is the scatter ploting of the correlation with the label varibale MEDV
from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()
#now from scatter plot we have found the interesting pattern inthe RM with MEDV and LSTAT with the MEDV
#so we here ploting the plot between MEDV and RM to find the outlier
housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)#to find the outlier between them and deleted the outlier from the dataset
plt.show()
#Trying out attribute combination
#so here we can see the tax and and rroms so we can relate this tax per room we can added the new attribute
housing["TAX per ROOM"]=housing["TAX"]/housing["RM"]
#now see the dataset
print(housing.head())
#now check the correlation of the matrix now with the label varibale MEDV
corr_matrix=housing.corr()
corr_matrix["MEDV"].sort_values(ascending=False)
#now plot the sactter plot of the new attribute col tax per room with the label variable MEDV
housing.plot(kind="scatter",x="TAX per ROOM",y="MEDV",alpha=0.8)
plt.show()
#split the data set into the indepenedent and dependent variable from the train dataset
housing=strat_train.drop("MEDV",axis=1)
housing_labels=strat_train["MEDV"].copy()

#Missing attribute
#To take care missing attribute we have the three option
#1.>Get rid of the missing values
#2.>Get rid of the whole values means drop the whole column
#3.>set the value to some values(fill with zero,fill with median,fill with the mean)
#op1.get rid of the missing values:- a=housing.dropna(subset=["RM"])
#op2.get rid of the entire column housing.drop("RM",axis=1)
#op3.1 set the new values
#we can use fillna method
#housing["RM"]=housing["RM"].fillna(housing["RM"].median())
#OP3.2
# we have to fill both train and test dataset cz in test dataset will have missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)
print(imputer.statistics_)
x=imputer.transform(housing)
#creating a new datahframe with all the values
housing_tr=pd.DataFrame(x,columns=housing.columns)
print(housing_tr.head())
print(housing_tr.describe())
#now adding values to the train data set
print(housing_tr.isna().sum())
"""
Scikit learn Design
"""
#before we trained our model if we want to build a good model we have to scale down our values in a same range to pwerform good
#we use feature scalling
#primarily two types of features scaling method
#Min-Max scvaling(normalization){(values-min)/max-min} o to 1 lies values
#Standardrization {values-mean)/std
#now creating a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ("Imputer",SimpleImputer(strategy="median")),
    ("std_scaler",StandardScaler())
])
#(we have done on the training set of the data)after scaling down values and impurte the values by the pipeline structure
housing_num_tr=my_pipeline.fit_transform(housing)
print("checking the standard scaler------")
print(housing_num_tr)
print(housing_num_tr.shape)
#selecting the right model for our problem
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
#for some prediction with some data from the main data frame
some_data=housing.iloc[:5]#iloc[:5,:5]means 5 row and 5 col only 5 rows with every feature column
some_labels=housing_labels.iloc[:5]
#now preprocessed the some data using pipeline
preprepare_data=my_pipeline.transform(some_data)
print(model.predict(preprepare_data))
print(list(some_labels))
#now evaluate our model with the mean_squared_error
from sklearn.metrics import mean_squared_error
import numpy as np
housing_predict=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predict)
lin_rmse=np.sqrt(lin_mse)
print(lin_rmse)#here we can see the in train set over fitting is done by the model bcz of rmse is 0
#using better technique-cross validation
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring='neg_mean_squared_error',cv=5)
rmse_score=np.sqrt(-scores)
#create functions to see the scores of the model
def socres_model(score):
    print("score of the model: ",score)
    print("mean scores of the model: ",score.mean())
    print("standard deviation of the model: ",score.std())
socres_model(rmse_score)
#now saving the model
#from joblib import dump,load
#dump(model,"Dragon.joblib")
#now testing the model
x_test=strat_test.drop("MEDV",axis=1)
y_test=strat_test["MEDV"]
X_test_prepared=my_pipeline.transform(x_test)#preparing the features of the dataset
final_prediction=model.predict(X_test_prepared)
final_mse=mean_squared_error(y_test,final_prediction)
final_rmse=np.sqrt(final_mse)
print(final_prediction,list(y_test))
print(preprepare_data[0])
#using the model
#from joblib import dump,load
#import numpy as np
#model=load("Dragon.joblib")
#now we have 1d array of prepare data we have to convert the 1d array to 2d array
#input=np.array([[-0.43942006 , 3.12628155, -1.12165014, -0.27288841, -1.42262747, -0.24583917,
 #-1.31238772 , 2.61111401, -1.0016859,  -0.5778192,  -0.97491834, 0.41164221,
# -0.86091034]]
#)
#print(model.predict(input))
#save and using the model
import pickle
pickle.dump(model,open('dragon_realestate.pkl','wb'))
model=pickle.load(open('dragon_realestate.pkl','rb'))

