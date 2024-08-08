import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.io
import numpy as np
import pandas as pd

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/nanoGPT
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775

model_config = 'ECG_PT' #switch between 'ECG_PT' and 'PPG_PT'

block_size = 500 # this is context length
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
model_path_ppg = "D:/HeartGPTModels/PPGPT_500k_iters.pth"
model_path_ecg = "D:/HeartGPTModels/ECGPT_560k_iters.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_config == 'PPG_PT':
    vocab_size = 102 #102 for PPGPT, 101 for ECGPT
    model_path = model_path_ppg
    context_path = 'D:/HeartGPTModels/example_context_PPG.csv'
elif model_config == 'ECG_PT':
    vocab_size = 101
    model_path = model_path_ecg
    context_path = 'D:/HeartGPTModels/example_context_ECG.csv'


def tokenize_biosignal(data):

    # Get the shape of the data
    shape = data.shape

    # If the data is a column vector, reshape it to a row vector
    if len(shape) > 1 and shape[0] > shape[1]:
        data = data.T

    # If there are more than 500 data points, select the last 500
    if data.shape[1] > 500:
        data = data[:, -500:]

    # Scale the values between 0 and 1
    data_min = np.min(data)
    data_max = np.max(data)
    data_scaled = (data - data_min) / (data_max - data_min)

    # Multiply by 100
    data_scaled *= 100

    # Round to the nearest integer
    data_rounded = np.round(data_scaled)

    return data_rounded

#model definition
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size)))) #buffer means not updated by optimiser
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention weights
        wei = q @ k.transpose(-2, -1) * C**-0.5 # square root headsize # (B, T, C) @ (B, C, T) = B, T, T
        # for every batch, we will now have a T by T matrix giving us the affinities of each token
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        # the tril signifies a decoder block, future tokens cannot communicate with the past
        wei = F.softmax(wei, dim=-1)# weights corresponding to the update of each token sum to 1

        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
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


# define the main heart_GPT model class
class Heart_GPT_Model(nn.Module):

    def __init__(self):
        super().__init__()

        # table needs to be vocab size by vocab size, to look up probability of next token given this token
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
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
        #channel is vocab size, so in this case 102 or 101

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

model = Heart_GPT_Model()

model.load_state_dict(torch.load(model_path))
model.eval()
m = model.to(device)

# load in context
#if it is PPG, make sure it is 50Hz sample frequecy. If ECG, 100Hz.

# Load the CSV file into a DataFrame
df = pd.read_csv(context_path, header=None)
# Convert the DataFrame to a numpy array
data = df.values

data_tokenised = tokenize_biosignal(data)
example_context_tensor = torch.tensor(data_tokenised, dtype=torch.long, device = device)


# now prompt the model with the context
print('Generating new tokens')
output = (m.generate(example_context_tensor, max_new_tokens=500)[0].tolist())

# convert output to DataFrame and save as csv
output_df = pd.DataFrame(output)
output_df.to_csv('D:/HeartGPTModels/model_output.csv', index=False, header=False)

data_tokenised = np.transpose(data_tokenised).tolist()
# convert data_tokenised to DataFrame and save as csv
data_tokenised_df = pd.DataFrame(data_tokenised)
# convert the dataframe to integer
data_tokenised_df = data_tokenised_df.astype(int)
data_tokenised_df.to_csv('D:/HeartGPTModels/tokenised_context.csv', index=False, header=False)
print('Generation saved to CSV')



