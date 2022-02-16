# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    x=(x-np.min(x))/100
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

Xtrain=feature_normalize(Xtrain)
plt.hist(Xtrain, density=True);
plt.title('normalize_Train')
plt.show()
Xtest=feature_normalize(Xtest)
plt.hist(Xtest, density=True);
plt.title('normalize_Test')
plt.show()
for feature, featurename in enumerate(columns[:-1]):
    print(feature+1,featurename)
    x_train = Xtrain[:, feature]
    x_test = Xtest[:, feature]
    
    #data normalize
    #x_train=feature_normalize(x_train)
    #x_test=feature_normalize(x_test)
    m=uni_linear(0,0,x_train,Ytrain,0.000001)
    print('m :',m[0],'b :',m[1])
    mtrain=MSE(m[0],m[1],x_train,Ytrain)
    mtest=MSE(m[0],m[1],x_test,Ytest)
    print('mse_train :',mtrain)
    print('mse_test :',mtest)
    print('variance_train',1-mtrain/variance(Ytrain))
    print('variance_test',1-mtest/variance(Ytest))
    
    print(variance(x_train))
    print(variance(x_test))
#    plt.title(featurename)
#    plot(m[0],m[1],x_train,Ytrain)