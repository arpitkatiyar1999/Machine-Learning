#importlibrary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dataset
dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values
# work on missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
X[:,1:3]=imputer.fit_transform(X[:,1:3])
# encode categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
label_X=LabelEncoder()
X[:,0]=label_X.fit_transform(X[:,0])
ct=ColumnTransformer([("Country",OneHotEncoder(),[0])], remainder = 'passthrough')
X=ct.fit_transform(X)
label_Y=LabelEncoder()
Y=label_Y.fit_transform(Y)

#splitting the dataset to training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)