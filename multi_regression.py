import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
#import data change the path to your file
path = input("enter filepath: ")
data =pd.read_excel(path)

columns = data.columns
train = np.array(data[:900])
test = np.array(data[900:])
Xtrain, Ytrain = train[:, :-1], train[:, -1]
Xtest, Ytest = test[:, :-1], test[:, -1]

scaler = StandardScaler()
#normalization 
Xtrain= scaler.fit_transform(Xtrain)
Xtest= scaler.fit_transform(Xtest)


def variance(x):
    xm=np.mean(x)
    n=len(x)
    total=0
    for i in range(n):
        sq=xm-x[i]
        total+=sq**2
    return total/n

def MSE(X, w, b, y):
    y_pred= X.dot(w) + b
    n = len(y)
    sum = 0
    for i in range(n):
        diff = y[i] - y_pred[i]
        sum += diff**2
    return sum/n



def GradientDescent(x, y, m, b, learning_rate, epochs):
    for epoch in range(epochs):
        ypred = x.dot(m) + b
        loss = ypred - y
        
        Dm = x.T.dot(loss) / len(y)
        Db = np.sum(loss) / len(y)
        
        m = m - learning_rate*Dm
        b = b - learning_rate*Db
  
        
    return m, b
#berfore normalzation: learning rate is 0.000001, m=0.  after normalzation learning rate =0.1 
m, b= GradientDescent(Xtrain, Ytrain, np.zeros(Xtrain.shape[1]), 5, 0.1,epochs=22000)
mse_train=MSE(Xtrain, m, b, Ytrain)
mse_test=MSE(Xtest, m, b, Ytest)
print(m)
print('variance_train',1-mse_train/variance(Ytrain))
print('variance_test',1-mse_test/variance(Ytest))
print('mse_train :',mse_train,'mse_test :',mse_test)

