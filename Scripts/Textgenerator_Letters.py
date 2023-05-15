import numpy as np
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import LSTM, Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import pickle

TRAINING = True
WINDOW_SIZE = 80
PREDICTION_LENGTH = 200

with open(r'..\data\Textgernerierung\verwandlung.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    #print(content)

# get the part that contains the story without discllaimer etc

content = "\n".join(content.split('\n')[59:1952])
# print(content)

unique_chars = set(content)

X = []
Y = []
for i in range(len(content) - WINDOW_SIZE):
    X.append(content[i: i + WINDOW_SIZE])
    Y.append(content[i + WINDOW_SIZE])

if TRAINING:
    char_to_index = {}
    index_to_char = {}
    for ix, char in enumerate(unique_chars):
        char_to_index[char] = ix
        index_to_char[ix] = char

    with open(r'..\models\Textgenerator\index_to_char.pickle', 'wb') as file:
        pickle.dump(index_to_char, file)
    with open(r'..\models\Textgenerator\char_to_index.pickle', 'wb') as file:
        pickle.dump(char_to_index, file)

    # One hot encode letters

    X = [[char_to_index[char] for char in X[i]] for i in range(len(X))]
    Y = [char_to_index[char] for char in Y]
    X = np.array(X).reshape(-1, WINDOW_SIZE, 1)

    X = to_categorical(X, num_classes=len(char_to_index))
    Y = to_categorical(Y, num_classes=len(char_to_index))

    print(X.shape)
    print(Y.shape)

    input = Input(shape=(WINDOW_SIZE, len(unique_chars)))
    lstm = LSTM(128, input_shape=(WINDOW_SIZE, len(unique_chars)), return_sequences=True)(input)
    lstm = Dropout(0.15)(lstm)
    lstm = LSTM(64, input_shape=(WINDOW_SIZE, len(unique_chars)))(lstm)
    lstm = Dropout(0.15)(lstm)
    output = Dense(len(unique_chars), activation='softmax')(lstm)

    model = Model(input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    save_model = ModelCheckpoint('..\\models\\Textgenerator\\weights_multilayer.{epoch:02d}-{loss:.2f}.hdf5')
    history = model.fit(X, Y, batch_size=64, epochs=10, callbacks=save_model)
    model.save_weights('..\\models\\Textgenerator\\model_weights_multilayer.hdf5')

else:
    with open(r'..\models\Textgenerator\char_to_index.pickle', 'rb') as file:
        char_to_index = pickle.load(file)
    with open(r'..\models\Textgenerator\index_to_char.pickle', 'rb') as file:
        index_to_char = pickle.load(file)

    input = Input(shape=(WINDOW_SIZE, len(unique_chars)))
    lstm = LSTM(128, input_shape=(WINDOW_SIZE, len(unique_chars)), return_sequences=True)(input)
    lstm = Dropout(0.15)(lstm)
    lstm = LSTM(64, input_shape=(WINDOW_SIZE, len(unique_chars)))(lstm)
    lstm = Dropout(0.15)(lstm)
    output = Dense(len(unique_chars), activation='softmax')(lstm)
    model = Model(input, output)
    model.load_weights('..\models\Textgenerator\model_weights_multilayer.hdf5')
    model.summary()

    # One hot encode letters
    X = [[char_to_index[char] for char in X[i]] for i in range(len(X))]
    Y = [char_to_index[char] for char in Y]
    X = np.array(X).reshape(-1, WINDOW_SIZE, 1)

    X = to_categorical(X, num_classes=len(unique_chars))
    Y = to_categorical(Y, num_classes=len(unique_chars))

test_char = content[100:100 + WINDOW_SIZE]
test_char_inx = [char_to_index[char] for char in test_char]
test_char_ohe = to_categorical(test_char_inx, num_classes=len(char_to_index))

print(test_char, end="")

for i in range(PREDICTION_LENGTH):
    ypred_prob = model.predict(test_char_ohe.reshape(1, WINDOW_SIZE, -1), verbose=0)
    index_pred = np.random.choice(68, 1, p=ypred_prob.reshape(-1))[0] # probabilistic generator to avoid deterministic loops
    letter_pred = index_to_char[index_pred]
    test_char = test_char + letter_pred
    test_char_ohe = test_char_ohe[1:, :]
    test_char_ohe = np.vstack([test_char_ohe, to_categorical(index_pred, num_classes=len(char_to_index))])
    print(letter_pred, end="")

print(test_char)

ypred_prob = model.predict(X[0].reshape(1, WINDOW_SIZE, -1))
ypred = [index_to_char[np.argmax(y)] for y in ypred_prob]

X_text = X[0]