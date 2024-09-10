import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Regression:
    def __init__(self,x,y,lr,degree = 1,regularization = 0,ld = 0):
        self.x = x.to_numpy().reshape(-1,1)
        self.y = y.to_numpy()
        i = 2

        while (i<=degree):
            xs = self.x[:,0]**i
            self.x = np.concatenate((self.x, xs.reshape(-1,1)), axis=1)
            i+=1

        # self.x = x
        self.reg = regularization
        self.deg = degree
        self.b = 0
        self.w= np.zeros(degree)
        self.lr = lr
        self.ld = ld
    def split_dataset(self):
        n = len(self.x)
        train = int(0.8*n)
        validate = int(0.1*n)
        test = n - train - validate
        x_train = self.x[0:train]
        x_validate = self.x[train:train+validate]
        x_test = self.x[train+validate:n]

        y_train = self.y[0:train]
        y_validate = self.y[train:train+validate]
        y_test = self.y[train+validate:n]
        
        return x_train,x_test,x_validate,y_train,y_test,y_validate
    
    def fit_train(self,x_train,y_train):
        self.x = x_train
        self.y = y_train
    
    def error(self,x_test,y_test):
        y_pred = np.dot(x_test, self.w) + self.b
        error = np.mean((y_test - y_pred) ** 2)
        return error

    def grad_desc(self):
        n = (self.x.shape[0])
        y_pred = np.dot(self.x, self.w) + self.b
        
        # Calculate gradients for weights and bias
        wg = (-2 / n) * np.dot(self.x.T, (self.y - y_pred))
        bg = (-2 / n) * np.sum(self.y - y_pred)
        if(self.reg==1):
            wg = (-2 / n) * np.dot(self.x.T, (self.y - y_pred))+ self.ld * np.sign(self.w)
            # bg = (-2 / n) * np.sum(self.y - y_pred)
        if(self.reg==2):
            wg = (-2 / n) * np.dot(self.x.T, (self.y - y_pred))+ 2 * self.ld * self.w
            # bg = (-2 / n) * np.sum(self.y - y_pred)



        # Update weights and bias
        self.w -= self.lr * wg
        self.b -= self.lr * bg
        
        # return self.w, self.b

    def prediction(self,x_pred):
        y_pred = np.dot(x_pred, self.w) + self.b
        return y_pred

    def metrics(self,x_pred,y_t):
        y_pred = np.dot(x_pred, self.w) + self.b
        mse = np.mean((y_t - y_pred) ** 2)
        std_dev = np.std(y_pred)
        variance = np.var(y_pred)        
        return mse,std_dev,variance
    
    def returnwandb(self):
        return self.w, self.b