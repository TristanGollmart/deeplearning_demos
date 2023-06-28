import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

# hyperparameters
batch_size = 4
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32

max_iters = 5000
eval_interval = 500
lr = 1e-3
eval_iters = 200

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

class FeedForward(nn.Module):
    # simple feed forward neural net
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # from paper porject 512 -> 2048 dimensions
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    # implements a single Head of self-Attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # information about token
        self.query = nn.Linear(n_embd, head_size, bias=False)  # what information token is looking for
        self.value = nn.Linear(n_embd, head_size, bias=False)  # passed value of token
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (Batch, Block, Headsize)
        q = self.query(x) # self attention: get query from same input instead of external source
        v = self.value(x)
        # interaction score of queries and keys
        wei = q @ k.transpose(-2, -1) # (B,T,16) @ (B,16,T) ---> (B,T,T): (b, i, j) = interaction of point i with point j
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) s.t. softmax(wei) = lower Triangular
        wei = F.softmax(wei, dim=-1)  # make it an interaction distribution: sum_j(w_ij) = 1 for all i
        return wei @ v # (T,T) @ (B, T, C) -> (B), (T, C) = (B, T, C)


class MultiHeadAttention(nn.Module):
    # implements multi-head attention
    # by concatenation of single head results in channel dimension
    def __init__(self, head_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class Block(nn.Module):
    # implements one block of multi-head self-attention followed by "in-node" postprocessing
    def __init__(self, n_embd, n_head):
        super.__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(head_size, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # pre-norm formulation, unlike the original paper!
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # self attention, followed by feed forward nn with skip connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# BigramLanguageModel
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # just get token embedding from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd/4)  # 4 self attention heads of embedding dimension 32/4 = 8
        self.lm_head = nn.Linear(n_embd, vocab_size)  # language model: maps token embeddings to logits to predict next letter/ word
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )

    def forward(self, idx, targets=None):
        # embeds the (B,T)-Tensor into a (B,T,C) tensor, C is analog of channels in ConvNets
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        x = self.blocks(x)

        logits = self.lm_head(x) # (B, T, vocab_size), decodes the encoded values <wei*v> of the self-attention block
        # for learning embedding: measure loss of prediction of next character from look up table compared to real sequence <target>

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # shape that cross entropy expects
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # generates max_new_tokens new tokens from idx
        # idx: (B, T) array
        for _ in range(max_new_tokens):
            # crop idx to the last block size tokens, so that items stay within scope of positional embedding layer
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # focus on last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # sample from this distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
# embed vector of size vocab_size (65), where each entry represents probability for next character:
# most simple: ohe: "a" of 100% -> 0 -> (1, 0, ...., 0, 0)
# very simple embedding since does not take into account
# any interactions between tokens: "a" is just "a" disregarding previous letters





B, T, C = batch_size, block_size, n_embd


# ------------------------ TRAIN MODEL -------------------------
print("start Training")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = eval_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # batch of data
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    # print(logits.shape)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate text from trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].to_list()))