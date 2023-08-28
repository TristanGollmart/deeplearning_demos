import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Input, Dropout, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.text import  one_hot, text_to_word_sequence
import pickle
import requests
import gensim
import os
from tqdm import tqdm
import math
import zipfile

# ------ get data ------
if not os.path.exists(r'..\models\Textgenerator\german.model'):
    url = "https://downloads.codingcoursestv.eu/037%20-%20neuronale%20netze/german.model.zip"
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))

    block_size = 1024

    with open(r'..\models\Textgenerator\german.model.zip', 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                         unit_divisor=1024, unit_scale=True):
            f.write(data)

    # Extracting .zipFile
    with zipfile.ZipFile(r"..\models\Textgenerator\german.model.zip", "r") as zipf:
        zipf.extract(r'..\models\Textgenerator\german.model')

    # Remove .zip-File (we don't need it anymore)
    os.remove(r"..\models\Textgenerator\german.model.zip")

    print("Done!")
else:
    print("Datei existiert bereits")



# ------------------------------
# Testing vector space embedding
# -----------------------------

sentence = "Das ist ein Satz. Ein Satz ist das"
# ohe function not good since hashing not unique
ohe_seq = text_to_word_sequence(sentence)
words = set(ohe_seq)

word_to_int = {}
int_to_word = {}
for k, v in enumerate(words):
    word_to_int[v] = k
    int_to_word[k] = v

words_tokens = [word_to_int[w] for w in words]
word_embedding = to_categorical(words_tokens)
print(word_embedding)

# -------------- TEST FINISH ----------------