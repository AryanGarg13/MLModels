import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys

curr_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(curr_dir, '../../models'))
sys.path.append(parent_dir)
print(curr_dir)

class KNN:
    def __init__(self,k,p=0,metric = 'euclidean'):
        self.k = k
        self.p = p
        self.metric = metric

    def df_numpy(self,df):
        df_n = df.to_numpy()
        return df_n 

    def shorten_df(self,df,sample_size = 1000):
        return df[0:sample_size]

    def split_dataset(self,X):

        X = X.sample(frac=1,random_state = 123).reset_index(drop=True)
        sample_size = X.shape[0]
        train = int(sample_size*0.8)
        test = int(sample_size*0.1)
        validate = sample_size-train-test
        x_train = X[0:train]
        x_test = X[train:train+test]
        x_validate = X[train+test:train+test+validate]
        return x_train,x_test,x_validate
    
    def set_distances(self,x_train,sample):
        if(self.metric == 'euclidean'):
            distance = np.sqrt(np.sum((x_train - sample)**2,axis=1))
            
        if(self.metric == 'cosine'):
            similarity = np.dot(x_train, sample) / (np.linalg.norm(x_train, axis=1) * np.linalg.norm(sample))
            distance = 1 - similarity
        if(self.metric == 'manhattan'):
            distance = np.sum(abs(x_train - sample),axis=1)
        if(self.metric == 'minkowski'):
            distance = (np.sum(abs(x_train - sample)**self.p,axis=1))**(1/self.p)

        return distance

    def predict(self,x_train,x_test,y_train):
        predictions = []
        for index,row in enumerate(x_test):
            # print(index)
            distance = self.set_distances(x_train,row)
            indices = np.argsort(distance)[:self.k]
            predictions.append(np.bincount(y_train[indices].astype(int)).argmax())
        return predictions


    def normalization(self, x_train, x_test, x_validate):
        dropped_train = x_train.drop(columns='artists')
        dropped_test = x_test.drop(columns='artists')
        dropped_validate = x_validate.drop(columns='artists')
        
        std = dropped_train.std()
        mean = dropped_train.mean()

        x_trnorm = (dropped_train-mean)/std
        x_tenorm = (dropped_test-mean)/std
        x_vanorm = (dropped_validate-mean)/std



        missing_train = x_train['artists']
        missing_test = x_test['artists']
        missing_validate = x_validate['artists']

        x_trnorm = pd.concat([x_trnorm, missing_train], axis=1)  
        x_tenorm = pd.concat([x_tenorm, missing_test], axis=1)  
        x_vanorm = pd.concat([x_vanorm, missing_validate], axis=1)  

        return x_trnorm,x_tenorm,x_vanorm

