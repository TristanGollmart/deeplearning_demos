from keras.utils import to_categorical
import math
import os
import zipfile
import nltk

import requests
from gensim.models import KeyedVectors
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from tqdm import tqdm

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
wv = KeyedVectors.load_word2vec_format("german.model", binary=True)
wv["auto"].shape
# Obama - USA + Russland = Putin
print(wv.most_similar(positiv=["Obama", "Russland"], negative=["USA"]))

# use KeyedVectors as keras embedding layer
from keras.models import Sequential
model = Sequential()
model.add(wv.get_keras_embedding())

with open("..\..\data\TextGenerator\verwandlung.txt", mode='r', encoding='utf-8') as f:
    content = f.read()

contents = "\n".join(content.split("\n")[59:1952])

