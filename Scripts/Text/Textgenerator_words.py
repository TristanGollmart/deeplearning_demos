import keras
from keras.utils import to_categorical
import math
import os
import zipfile
import numpy as np
import nltk
from nltk import word_tokenize

import requests
from gensim.models import KeyedVectors
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer


# ------ get file ------
if not os.path.exists(r'..\..\models\Textgenerator\german.model'):
    url = "https://downloads.codingcoursestv.eu/037%20-%20neuronale%20netze/german.model.zip"
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))

    block_size = 1024

    with open(r'..\..\models\Textgenerator\german.model.zip', 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                         unit_divisor=1024, unit_scale=True):
            f.write(data)

    # Extracting .zipFile
    with zipfile.ZipFile(r"..\..\models\Textgenerator\german.model.zip", "r") as zipf:
        zipf.extract(r'..\..\models\Textgenerator\german.model')

    # Remove .zip-File (we don't need it anymore)
    os.remove(r"..\..\models\Textgenerator\german.model.zip")

    print("Done!")
else:
    print("Datei existiert bereits")


# ------------------------------
# Testing vector space embedding
# -----------------------------

'''
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


# embedding word to N-dim embedding space (default 300)
wv = KeyedVectors.load_word2vec_format(r'..\..\models\Textgenerator\german.model', binary=True)
print(wv["auto"].shape)
# Obama - USA + Russland = Putin
print(wv.most_similar(positive=["Obama", "Russland"], negative=["USA"]))
'''

# Test on usecase -> verwandlung by Kafka
with open(r"..\..\data\TextGenerator\verwandlung.txt", mode='r', encoding='utf-8') as f:
    content = f.read()

contents = "\n".join(content.split("\n")[59:1952])

# tokenize input
# nltk.download("punkt")
tokens = word_tokenize(contents)
print(tokens[1:5])

# for performance: get top 1000 words
cv = CountVectorizer(max_features=1000, lowercase=False, token_pattern='(.*)')
cv.fit(tokens)
features = cv.get_feature_names()

word_to_int = {}
int_to_word = {}
for k, v in enumerate(features):
    word_to_int[v] = k
    int_to_word[k] = v
k_new = len(word_to_int)

def encode(word_list):
    #tokens = word_tokenize(mytext)
    i_list = []
    for w in word_list:
        if w in word_to_int:
            i_list.append(word_to_int[w])
        else:
            pass
    return i_list

def decode(lst):
    words = []
    for i in lst:
        if i in int_to_word:
            words.append(int_to_word[i])
        else:
            pass
    return "".join(words)

encoded = encode(tokens) #[word_to_int[w] if w in word_to_int else k_new for w in tokens]
vocab_size = max(encoded) + 1

# transform to sequence
block_size = 40
embed_dim = 100

x = []
y = []
for i in range(block_size, len(encoded)):
    x.append(encoded[i-block_size:i])
    y.append(encoded[i])

y_ohe = to_categorical(y, num_classes=vocab_size)

x = np.array(x)
y_ohe = np.array(y_ohe)
y = np.array(y)

# >> NO EMBEDDING SINCE MODEL WILL HANDLE THIS INTERNALLY WITH AN EMBEDDING LAYER
# Build model
from keras.layers import Embedding, LSTM, Dense
import tensorflow as tf
from keras.activations import softmax

class TextGenerator(keras.Model):
    def __init__(self, vocab_size, n_embed, output_dim, sequence_length, lstm_units=64):
        super(TextGenerator, self).__init__()
        self.n_embed = n_embed
        self.sequence_length = sequence_length
        self.embedding = Embedding(vocab_size, n_embed, input_length=self.sequence_length)
        self.lstm1 = LSTM(lstm_units, activation="tanh", return_sequences=True) # [B, T, C]
        self.fc1 = Dense(lstm_units, activation="relu")
        self.lstm2 = LSTM(lstm_units, activation="tanh", return_sequences=True)
        self.lstm3 = LSTM(lstm_units, activation="tanh", return_sequences=False) # [B, C]
        self.logits = Dense(output_dim, activation=None) # [B, C2]

    def call(self, inputs):
        x = self.embedding(inputs)  # B,T -> B,T,C
        x0 = self.fc1(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = x + x0          # here sequence returned : dimensions match: B, T, C
        x = self.lstm3(x)
        logits = self.logits(x) # possibly add softmax
        return logits

    def generate(self, input, n_words):
        '''
        input [1, T, C]
        output [n_words, C]
        :param input:
        :param n_words:
        :return:
        '''
        words = []
        for _ in range(n_words):
            inputs_cond = input[-self.sequence_length:]
            logits = self(tf.expand_dims(inputs_cond, 0))
            probs = softmax(logits)
            iword = tf.squeeze(tf.random.categorical(logits, 1))

            input = tf.concat([input, tf.expand_dims(iword, axis=0)], axis=0)
            words.append(decode([iword.numpy()]))
        return words

model = TextGenerator(vocab_size=vocab_size, n_embed=embed_dim, output_dim=vocab_size, sequence_length=block_size)
if os.path.exists(r'..\..\models\Textgenerator\kakfka_model_weights.index'):
    model.load_weights(r'..\..\models\Textgenerator\kakfka_model_weights.index')

model.compile(optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics="accuracy") #[keras.metrics.SparseCategoricalAccuracy(name='acc')])

# test_input = tf.expand_dims([np.random.randint(0, vocab_size) for _ in range(block_size)], axis=0)
# test_input =np.array([[np.random.randint(0, vocab_size) for _ in range(block_size)],
#              [np.random.randint(0, vocab_size) for _ in range(block_size)]])
# test_prediction = model.predict(test_input)
# print(test_prediction.shape)
# i = np.argmax(test_prediction)
# print(int_to_word[i])
#
# cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# print("first loss: ", cce(y_ohe, model.predict(x)).numpy())


# Test generator
generated = model.generate(x[0], 100)
print(" ".join(generated))
# Test end

history = model.fit(x, y, epochs=10, validation_split=0.2)

generated_text = model.generate(x[0], 100)
print("generating: ", " ".join(generated_text))
model.save_weights(r'..\..\models\Textgenerator\kakfka_model_weights')

print(history)