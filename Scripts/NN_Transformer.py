import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 4
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
# ----------------

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

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)

# train test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(42)

# Example: split into chunks for statistical learning of different attention heads
# on blocks of all different length up to block_size

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
    def __init__(self):
        super().__init__()
        # just get token embedding from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # language model: maps token embeddings to logits to predict next letter/ word
    def forward(self, idx, targets):
        # embeds the (B,T)-Tensor into a (B,T,C) tensor, C is analog of channels in ConvNets
        B, T = idx.shape


        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        logits = self.lm_head(tok_emb) # (B, T, vocab_size)
        # for learning embedding: measure loss of prediction of next character from look up table compared to real sequence <target>
        B, T, C = logits.shape
        x = tok_emb + pos_emb # (B, T, C)
        logits = logits.view(B*T, C) # shape that cross entropy expects
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

m = BigramLanguageModel()
# embed vector of size vocab_size (65), where each entry represents probability for next character:
# most simple: ohe: "a" of 100% -> 0 -> (1, 0, ...., 0, 0)
# very simple embedding since does not take into account
# any interactions between tokens: "a" is just "a" disregarding previous letters

logits, loss = m(xb, yb)
print(logits.shape)


# calc average of preovious tokens
# use bow (bag of words) since just averaging, no weighted/ learnable connections
# like self attention
B, T, C = batch_size, block_size, n_embd
x = torch.randn(B, T, C)
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # shape (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

# here efficient implementation of above algebra using vectorization
# wei = (0,..., 0) : Bag of words. can also be learnable weights to encode interaction strength "self attention", just weighted aggregation

# single Head
head_size = 16
key = nn.Linear(C, head_size, bias=False)   # information about token
query = nn.Linear(C, head_size, bias=False)  # what information token is looking for
value = nn.Linear(C, head_size, bias=False)  # passed value of token
k = key(x) # (B,T,headsize)
q = query(x)
# interaction of queries and keys
wei = q @ k.transpose(-2, -1) # (B,T,16) @ (B,16,T) ---> (B,T,T)

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T)) #: no interaction
wei = wei.masked_fill(tril==0, float('-inf')) # s.t. softmax(wei) = lower Triangular
wei = F.softmax(wei, dim=-1)
v = value(x)
xbow2 = wei @ v # (T,T) @ (B, T, C) -> (B), (T, C) = (B, T, C)

print(np.isclose(xbow, xbow2, rtol=1e-4).all())