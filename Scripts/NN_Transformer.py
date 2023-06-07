
# read shakespeare dataset
with open(file=r"..\\data\\NN_Transformer\\tinyshakespeare.txt", encoding="utf-8") as f:
    text = f.read()

print("length of text: ", len(text))

# tokenize: build a mapping individual chars->numbers
# also possible to use whole words or subwords -> larger vocab size but shorter encoding, like tiktoken.get_encoding('gpt2')
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)

char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s]  # maps string to according integers
decode = lambda lst: "".join([int_to_char[i] for i in lst])  # maps list of integers to string

print(encode("hii there"))

# encode the whole input text
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)

# train test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]