import keras.backend as K
import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, GRU, Embedding, Bidirectional, BatchNormalization, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping
from keras.layers.merge import add, dot, concatenate
import numpy as np
import random
import pandas as pd
import pickle
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
train_caption_path = sys.argv[3]
test_caption_path = sys.argv[4]
output_path = sys.argv[5]

train_data = np.load(train_data_path)
test_data = np.load(test_data_path)
print(len(train_data),"個 training 音檔")
print(len(test_data),"個 testing 音檔")
max_frame_length = np.max([len(sample) for sample in train_data])
print("max langth of wav:",max_frame_length)

# load caption
with open(train_caption_path,"r") as f:
    train_caption = f.readlines()
    train_caption = [sent.strip() for sent in train_caption]
    train_sentences = [sent.split(" ") for sent in train_caption]
with open(test_caption_path,"r") as f:
    test_choice = f.readlines()
    test_choice = [sent.strip() for sent in test_choice]
    test_corpus = ",".join(test_choice)
    test_sentences = [sent.split(" ") for sent in test_corpus.split(",")]
    test_corpus = test_corpus.replace(",", " ")


# chinese character level tokenizer
tokenizer = Tokenizer(num_words=None,filters='\n', lower=True, split=" ", char_level=False)
tokenizer.fit_on_texts(train_caption + [test_corpus])
print("number of token in caption:", len(tokenizer.word_index))
inv_map = {v: k for k, v in tokenizer.word_index.items()}


train_caption_sequences = tokenizer.texts_to_sequences(train_caption)
max_length = np.max([len(i) for i in train_caption_sequences])
print("max length:", max_length)


test_caption_sequences =  tokenizer.texts_to_sequences([" ".join(sample) for sample in test_sentences])

# pad sequence
test_caption_pad = pad_sequences(test_caption_sequences, maxlen=max_length)
test_data_pad = pad_sequences(test_data, maxlen=max_frame_length,dtype='float32')
test_data_pad_expand = np.repeat(test_data_pad, 4,axis=0)
# revert
print(test_caption_pad.shape)
print(test_data_pad_expand .shape)

model_path = "../model"

model_names = ["model6_randomAug1_1V5_2layers_512_88.h5", "model6_randomAug1_1V5_2layers_512_92.h5",
                "model6_randomAug1_1V5_2layers_512_93.h5"]

p = []
for name in model_names:
    model = load_model(os.path.join(model_path,name))
    print(name)
    prediction = model.predict([test_data_pad_expand,test_caption_pad])
    p.append(prediction)
pred_y_prob = np.sum(p,axis = 0)
# load submit
sample_submit = pd.read_csv("sample_submission.csv")
pred_y = np.argmax(pred_y_prob.reshape(-1,4),axis=1)
sample_submit["answer"] = pred_y
sample_submit.to_csv(output_path,index=None)