import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Input,InputLayer,BatchNormalization, Dense, Bidirectional,LSTM,Dropout,GRU, Conv1D, MaxPool1D, Activation
from keras.models import Model,load_model
from keras.callbacks import History ,ModelCheckpoint
from keras import backend as K
import pandas as pd
import gensim
import random
import sys
import os
import re
import pickle
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
ensemble = False

test_data_path = sys.argv[1]
prediction_path = sys.argv[2]

with open(test_data_path) as f:
    test = f.readlines()


stemmer = gensim.parsing.porter.PorterStemmer()

def preprocess(string, use_stem = True):
    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")\
    .replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")\
    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")\
    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")\
    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")\
    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")\
    .replace("couldn ' t","couldnt")
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    return string


test_X = [preprocess("".join(sample.split(",")[1:])).strip() for sample in test[1:]]
test_X = [sent for sent in stemmer.stem_documents(test_X)]


vocab_size = None
# tokenizer = Tokenizer(num_words=vocab_size, filters="\n\t")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# tokenizer.fit_on_texts(train_X_cleaned + test_X + train_nolab_clean)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

max_length = 39
sequences_test = tokenizer.texts_to_sequences(test_X)
test_X_num = pad_sequences(sequences_test, maxlen=max_length)


if ensemble == True:
	model1 = load_model("./model3/modelW2V-00005-0.83130.h5")
	model2 = load_model("./model3/modelW2V-00003-0.83730.h5")
	model3 = load_model("./model3/modelW2V-00005-0.83558.h5")
	model4 = load_model("./model3/modelW2V-00004-0.83537.h5")
	model5 = load_model("./model3/modelW2V-00004-0.82980.h5")
	model6 = load_model("./model3/modelW2V-00003-0.83340.h5")
	model7 = load_model("./model3/modelW2V-00002-0.82935.h5")
	model8 = load_model("./model3/modelW2V-00003-0.82980.h5")
	model9 = load_model("./model3/modelW2V-00003-0.82558.h5")
	model10 = load_model("./model3/modelW2V-00006-0.83301.h5")
	print("models loaded")
	p1 = model1.predict(test_X_num)
	p2 = model2.predict(test_X_num)
	p3 = model3.predict(test_X_num)
	p4 = model4.predict(test_X_num)
	p5 = model5.predict(test_X_num)
	p6 = model6.predict(test_X_num)
	p7 = model7.predict(test_X_num)
	p8 = model8.predict(test_X_num)
	p9 = model9.predict(test_X_num)
	p10 = model10.predict(test_X_num)
	# store some test_X_num for self training
	pred_y_prob = (p1+p2+p3+p4+p5+p6+p7+p8+p9+p10)/10
else:
	model = load_model("modelW2V-00005-0.83330.h5")
	print(model.summary())
	pred_y_prob = model.predict(test_X_num)
pred_y = np.argmax(pred_y_prob,axis=1)
submission = pd.read_csv("sampleSubmission.csv")
submission["label"] = pred_y
submission.to_csv(prediction_path,index=False)
print("submission file saved")