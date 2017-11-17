import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling2D,GlobalAveragePooling1D
from keras.layers import Flatten
from keras.models import load_model
from keras.layers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History ,ModelCheckpoint
from keras.layers import Activation, LeakyReLU
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import collections
import sys
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

## load data
def normalize_1(x):
    x = (x - x.mean())/x.std()
    return x
def normalize_2(x):
    x = x/255.
    return x
# load encoder to decode label
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

test_path = sys.argv[1]
output_path = sys.argv[2]

test = pd.read_csv(test_path)
test_X = np.array([row.split(" ") for row in test["feature"].tolist()],dtype=np.float32)
test_X = normalize_2(test_X.reshape(-1,48,48,1))


print("load model ...")
model = load_model("model/model1-00204-0.72588.h5")
print("prediting...")
p = model.predict(test_X)

pred_y = encoder.inverse_transform(p)

sample = pd.read_csv("sample.csv")
sample["label"] = pred_y
sample.to_csv(output_path,index=None)