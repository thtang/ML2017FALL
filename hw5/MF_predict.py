import keras.backend as K
from keras.regularizers import l2
import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, GRU, Embedding, Bidirectional, Flatten, Dropout, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, dot, concatenate, multiply, average
import numpy as np
from keras.layers import Dot
from keras.initializers import Zeros
import random
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from keras.engine.topology import Layer
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from keras.utils import get_custom_objects
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import pickle
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# read test data
test_path = sys.argv[1]
prediction_path = sys.argv[2]

test_data = pd.read_csv(test_path)
with open("user2id.pkl","rb") as f:
	user2id = pickle.load(f)

with open("movie2id.pkl","rb") as f:
	movie2id = pickle.load(f)

class WeightedAvgOverTime(Layer):
    """
    The code of this layer is from
        https://github.com/WindQAQ/ML2017/blob/master/hw6/train.py
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedAvgOverTime, self).__init__(**kwargs)
    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            s = K.sum(mask, axis=1)
            if K.equal(s, K.zeros_like(s)) is None:
                return K.mean(x, axis=1)
            else:
                return K.cast(K.sum(x * mask, axis=1) / K.sqrt(s), K.floatx())
        else:
            return K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, x, mask=None):
        return None

    def get_config(self):
        base_config = super(WeightedAvgOverTime, self).get_config()
        return dict(list(base_config.items()))
    
def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


test_user = np.array([user2id[i] for i in test_data["UserID"]])
test_movie = np.array([movie2id[i] for i in test_data["MovieID"]])

mean = 3.58171208604
get_custom_objects().update({"rmse": rmse,"mean":mean,"WeightedAvgOverTime":WeightedAvgOverTime})

model_1 = load_model("model_5.h5")
pred_rating1 = model_1.predict([test_user,test_movie])
sampleSubmission = pd.read_csv("SampleSubmisson.csv")
sampleSubmission["Rating"] = np.clip(pred_rating1,1,5)
sampleSubmission.to_csv(prediction_path,index=None)