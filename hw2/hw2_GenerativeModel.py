# coding: utf-8

import numpy as np
import pandas as pd
import csv
import sys


X_train_path = sys.argv[3]
Y_train_path = sys.argv[4]
X_test_path = sys.argv[5]
output_path = sys.argv[6]
print("load model")
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


class GenerativeModel(object):
    def __init__(self):
        a = None
    def fit(self, X,y):
        ####### find mean, coveriance matrix of two class ######
        # seperate y=0 and y=1
        self.c1_ind = np.where(y==0)[0]
        self.c2_ind = np.where(y==1)[0]
        
        # compute stastistic
        self.c1_X = X[c1_ind,:]
        self.c2_X = X[c2_ind,:]
        self.u1 = np.mean(c1_X ,axis=0)
        self.u2 = np.mean(c2_X ,axis=0)
        self.p1 = len(c1_ind)/(len(c1_ind)+len(c2_ind))
        self.p2 = 1 - p1
        self.cov =  np.cov(c1_X,rowvar=False) * p1 + np.cov(c2_X,rowvar=False) * p2
        print("use psudo inverse")
        self.cov_inv = np.linalg.pinv(cov)
        
    def predict(self, test_X):
        z =((test_X).dot(self.cov_inv).dot(self.u1-self.u2)- 
            (1/2)*(self.u1).dot(self.cov_inv).dot(self.u1)+ (1/2)*(self.u2).dot(self.cov_inv).dot(self.u2)
            +np.log(len(self.c1_ind)/len(self.c2_ind)))
        return self.sign(self.sigmoid(z))
    
    def sigmoid(self, z):
        res = 1 / (1.0 + np.exp(-z))
        return np.clip(res, 1e-8, 1-(1e-8))
    def sign(self, a):
        output = []
        for i in a:
            if i>0.5:
                output+="0"
            else:
                output+="1"
        return output

print("fitting....")
GM = GenerativeModel()
GM.fit(train_X,train_y)
pred_y = GM.predict(test_X)
print("save prediction")
sample = pd.read_csv("sample_submission.csv")
sample["label"] = pred_y
sample.to_csv(output_path, index = None)