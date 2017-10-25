# coding: utf-8

import numpy as np
import pandas as pd
import csv
import sys

X_train_path = sys.argv[3]
Y_train_path = sys.argv[4]
X_test_path = sys.argv[5]
output_path = sys.argv[6]



def sign(a):
        output = []
        for i in a:
            if i<0.5:
                output+="0"
            else:
                output+="1"
        return output
    
class LogisticRegression_ADA(object):

    def __init__(self, eta=1, n_iter=100, random_state=1, alpha=0, shuffle = False):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        self.alpha = alpha
    def fit(self, X, y):
        print(X.shape)
        
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.cost_ = []
        lr_w = np.zeros(X.shape[1])
        for i in range(self.n_iter):
            
            w_grad = np.zeros(X.shape[1])
            
            if self.shuffle:
                X, y = self._shuffle(X,y)
                
            for xi, target in zip(X,y): # iterate on single sample
                cost = []               # record cost for each sample
                output = self.sigmoid(self.net_input(xi))
                error = (target - output)
                w_grad = w_grad - 2*xi.dot(error)
                cost.append(error)
            lr_w = lr_w + w_grad**2
        
            self.w_ = self.w_ - self.eta/np.sqrt(lr_w) * (w_grad)

        
            # calculate RMSE for an epoch
            self.cost_.append(abs(np.average(cost)))
        return self
    
    def sigmoid(self,z):
        res = 1 / (1.0 + np.exp(-z))
        return np.clip(res, 1e-8, 1-(1e-8))

    def net_input(self, X):
        return np.dot(X, self.w_)

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
        return self.sigmoid(self.net_input(X))
    
    def _shuffle(self,X,y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

print("load data")
text = open(X_train_path, 'r') 
row = csv.reader(text , delimiter=",")

X_train = []
for r in row:
    X_train.append(r)
    

text = open(Y_train_path, 'r') 
row = csv.reader(text , delimiter=",")
Y_train = []
for r in row:
    Y_train.append(r)


text = open(X_test_path, 'r') 
row = csv.reader(text , delimiter=",")

X_test = []
for r in row:
    X_test.append(r)
        
    
    
train_X = (np.array(X_train)[1:,]).astype("float")
columns = np.array(X_train)[1,:]

train_y = np.array(Y_train[1:]).flatten().astype("float")
test_X = (np.array(X_test)[1:,]).astype("float")

X_mean = train_X[:,[0,1,3,4,5]].mean(axis=0)
X_std = train_X[:,[0,1,3,4,5]].std(axis=0)

train_X[:,[0,1,3,4,5]] = (train_X[:,[0,1,3,4,5]]- X_mean) / X_std
test_X[:,[0,1,3,4,5]] = (test_X[:,[0,1,3,4,5]]- X_mean) / X_std



print("fitting....")
logit = LogisticRegression_ADA()
logit.fit(train_X,train_y)


pred_y = sign(logit.predict(test_X))
print("save")
sample = pd.read_csv("sample_submission.csv")
sample["label"] = pred_y
sample.to_csv(output_path, index = None)
