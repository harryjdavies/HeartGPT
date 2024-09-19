import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.io
import numpy as np
import time

# Harry Davies 19_09_2024

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/ng-video-lecture
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775

eval_interval = 2000
save_interval = 20000 #how often the model is checkpointed
eval_iters = 200
batch_size = 64 # sequences we process in parellel
max_iters = 1000000
block_size = 500 # this is context length
learning_rate = 3e-04
n_embd = 64 # 384 / 6 means every head is 64 dimensional
n_head = 8
n_layer = 8
dropout = 0.2


# GPU is necessary. Training of 8 head, 8 layer model and 500 context length was possible with 12GB VRAM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
# data was loaded as a .mat file, but this is not necessary.
data_load = scipy.io.loadmat('D:/ecg_store_gpt_training')
data_ecg = data_load['ecg_store']

#define vocab size. All data was scaled between 0 and 100 and rounded to nearest integer, giving 101 possible token values
vocab_size = 101

perm = np.random.permutation(data_ecg.shape[0])
data_ecg_rand = data_ecg[perm,:]

#now time for some pytorch, convert to a torch tensor
data = torch.tensor(data_ecg_rand, dtype=torch.long)

# split so 90% for training, 10% for testing
x_thresh = int(0.9*data_ecg.shape[0])
train_data = data[:x_thresh,:]
test_data = data[x_thresh:,:]


def get_batch(split):
    data = train_data  if split == 'train' else test_data
    # creates two random indices. One to pick the subject, and one to pick the position in the trace.
    # traces for each subject were never less than 1000 samples. blocksize+ix2 can never be longer than the trace.
    ix = torch.randint(data.shape[0], (batch_size,))
    ix2 = torch.randint(500, (1,))
    x = torch.stack([data[i,ix2:ix2+block_size] for i in ix])
    y = torch.stack([data[i,ix2+1:ix2+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size)))) #buffer means not updated by optimiser
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #start = time.time()
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # square root headsize # (B, T, C) @ (B, C, T) = B, T, T
        # for every batch, we will now have a T by T matrix giving us the affinities of each token
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        # the tril signifies a decoder block, future tokens cannot communicate with the past
        wei = F.softmax(wei, dim=-1)# all attention weights sum to 1 for updating a single token
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        #end = time.time()
        #print(start-end)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        # creating a list of head objects (turned into modules) resulting in a number of head modules
        # then assigns the list of modules to self.heads - these run in parellel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) #projection generally matches sizes for adding in residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #concatenate the output of the different attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), #multiplication performed in attention is all you need paper
            # expands and contracts back down to projection
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # communication
        self.sa = MultiHeadAttention(n_head, head_size)
        # computation
        self.ffwd = FeedForward(n_embd)
        # layer norm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



# create heart GPT class
class HeartGPTModel(nn.Module):

    def __init__(self):
        super().__init__()
        # table needs to be vocab size by vocab size, to look up probability of next token given this token
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        #idx is batch, targets is time
        tok_emb = self.token_embedding_table(idx) #(B, T, vocab_size) which is batch, time, channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C (integers from 0 to T-1)
        x = tok_emb + pos_emb # B, T, C
        x = self.blocks(x) # B, T, C
        x = self.ln_f(x) # B, T, C

        logits = self.lm_head(x)
        #channel is vocab size, so in this case 65

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx (context) to the last block_size tokens because positional embeddings only has up to block size
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = HeartGPTModel()
m = model.to(device)
# random loss at this point would be -log(1/65)

#AdamW
optimizer  = torch.optim.AdamW(m.parameters(), lr=learning_rate)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# counter the number of model parameters to be trained
num_parameters = count_parameters(model)
print(f"The model has {num_parameters} trainable parameters.")

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if iter % save_interval == 0:
        #model_path for checkpointing
        model_path = f"D:/ECGPT_pretrained_{n_embd}_{n_head}_{n_layer}_{block_size}_{max_iters}_{iter}.pth"
        torch.save(model.state_dict(), model_path)
    #get batch
    x_batch, y_batch = get_batch('train')

    # loss evaluation
    logits, loss = m(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()







