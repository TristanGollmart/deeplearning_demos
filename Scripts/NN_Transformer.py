
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
import torch.nn as nn
from torch.nn import functional as F

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)

# train test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(42)

# Example: split into chunks for statistical learning of different attention heads
# on blocks of all different length up to block_size
block_size = 8
train_data[:block_size+1] # n+1 will lead to n chunks separate chunks of length 1,...,n: "h", "ha", "hal", ...
def create_chunks(x ,y):
    context = []
    target = []
    for t in range(block_size):
        context.append(x[:t+1])
        target.append(y[t])
    return context, target

context, target = create_chunks(train_data[:block_size], train_data[1:block_size+1])

print(context)
print(target)

# split set into batches
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # random batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# shapes of xb: [batch_size, block_size, ]
xb, yb = get_batch('train')
print(xb.shape, yb.shape)

# BigramLanguageModel
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # just get token embedding from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # embeds the (B,T)-Tensor into a (B,T,C) tensor, C is analog of channels in ConvNets
        logits = self.token_embedding_table(idx)
        # for learning embedding: measure loss of prediction of next character from look up table compared to real sequence <target>
        B, T, C = logits.shape
        logits = logits.view(B*T, C) # shape that cross entropy expects
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

m = BigramLanguageModel(vocab_size)
# embed vector of size vocab_size (65), where each entry represents probability for next character:
# most simple: ohe: "a" of 100% -> 0 -> (1, 0, ...., 0, 0)
# very simple embedding since does not take into account
# any interactions between tokens: "a" is just "a" disregarding previous letters

logits, loss = m(xb, yb)
print(logits.shape)