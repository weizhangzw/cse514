import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import data change the path to your file
data =pd.read_excel("C:\\Users\\zhang\\OneDrive\\桌面\\Concrete_Data.xls")

columns = data.columns
train = np.array(data[:900])
test = np.array(data[900:])
Xtrain, Ytrain = train[:, :-1], train[:, -1]
Xtest, Ytest = test[:, :-1], test[:, -1]

scaler = StandardScaler()
#normalization 
#Xtrain= scaler.fit_transform(Xtrain)
#Xtest= scaler.fit_transform(Xtest)


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



def GradientDescent(x, y, w, b, learning_rate, epochs):
    for epoch in range(epochs):
        ypred = x.dot(w) + b
        loss = ypred - y
        
        weight_gradient = x.T.dot(loss) / len(y)
        bias_gradient = np.sum(loss) / len(y)
        
        w = w - learning_rate*weight_gradient
        b = b - learning_rate*bias_gradient
  
        
    return w, b
#berfore normalzation: learning rate is 0.000001, w=0.  after normalzation learning rate =0.1 
w, b= GradientDescent(Xtrain, Ytrain, np.zeros(Xtrain.shape[1]), 0, 0.1,epochs=22000)
mse_train=MSE(Xtrain, w, b, Ytrain)
mse_test=MSE(Xtest, w, b, Ytest)
print('variance_train',1-mse_train/variance(Ytrain))
print('variance_test',1-mse_test/variance(Ytest))
print('mse_train :',mse_train,'mse_test :',mse_test)

