import numpy as np
import pandas as pd
import csv
from xgboost import XGBClassifier
import sys

X_train_path = sys.argv[3]
Y_train_path = sys.argv[4]
X_test_path = sys.argv[5]
output_path = sys.argv[6]
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
columns = np.array(X_train)[0,:]

train_y = np.array(Y_train[1:]).flatten().astype("float")
test_X = (np.array(X_test)[1:,]).astype("float")

depth = 6
r_lambda = 0
col_sample = 0.5
est = 100

print("fitting...")
XGB = XGBClassifier(max_depth = depth, reg_lambda = r_lambda,colsample_bytree = col_sample,n_estimators=est)
XGB.fit(train_X, train_y, eval_metric="auc",verbose=True)
XGB_pred = XGB.predict(test_X)
print("save prediction")
sample = pd.read_csv("sample_submission.csv")
sample["label"] = [str(pred)[0] for pred in XGB_pred]
sample.to_csv(output_path , index = None)