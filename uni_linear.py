# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
#import data change the path to your file
data =pd.read_excel("C:\\Users\\zhang\\OneDrive\\桌面\\Concrete_Data.xls")


columns = data.columns
train = np.array(data[:900])
test = np.array(data[900:])
Xtrain, Ytrain = train[:, :-1], train[:, -1]
Xtest, Ytest = test[:, :-1], test[:, -1]


def uni_linear(m,b,x,y,lr):
    Dm = 0
    Db = 0
    n = len(x)
    for i in range(10000):
        ypred = m*x + b  
        Dm = (-2/n) * sum(x * (y - ypred)) 
        Db = (-2/n) * sum(y - ypred)  
        m =m-  lr * Dm  
        b =m-  lr * Db
    return m,b

def variance(x):
    xm=np.mean(x)
    n=len(x)
    total=0
    for i in range(n):
        sq=xm-x[i]
        total+=sq**2
    return total/n

def feature_normalize(x):
    x=(x-np.mean(x))/np.max(x)
    return x

def plot(m,b,x,y):
    plt.scatter(x,y)
    Y_pred = m*x_train + b
    plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='red')
    plt.show()
def MSE(m,b,x,y):
    ypred=m*x+b
    n = len(y)
    sum = 0
    for i in range(n):
        diff = y[i] - ypred[i]
        sum += diff**2
    return sum/n

#plt.title('Before')
#plt.hist(Xtrain)
#plt.hist(Xtest)
#plt.show()

scaler = StandardScaler()
#Xtrain= scaler.fit_transform(Xtrain)
#Xtest= scaler.fit_transform(Xtest)
#plt.hist(Xtrain)
#plt.hist(Xtest)
#plt.title('After')
#plt.show()

for feature, featurename in enumerate(columns[:-1]):
    print(feature+1,featurename)
    x_train = Xtrain[:, feature]
    x_test = Xtest[:, feature]


    m=uni_linear(0,0,x_train,Ytrain,0.000001)
    print('m :',m[0])
    print('b :',m[1])
    mtrain=MSE(m[0],m[1],x_train,Ytrain)
    mtest=MSE(m[0],m[1],x_test,Ytest)
    print('mse_train :',mtrain)
    print('mse_test :',mtest)
    print('variance_train',1-mtrain/variance(Ytrain))
    print('variance_test',1-mtest/variance(Ytest))
#    plt.title(featurename)
#    plot(m[0],m[1],x_train,Ytrain)