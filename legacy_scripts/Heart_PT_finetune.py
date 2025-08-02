import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.io
import numpy as np

# Harry Davies 12_08_2024

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
elif model_config == 'ECG_PT':
    vocab_size = 101
    model_path = model_path_ecg


#model definition
class Head(nn.Module):

    def __init__(self, head_size, mask=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.mask = mask
        self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        if self.mask:
            wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, mask=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, mask=mask) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head, mask=True):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, mask=mask)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NewHead(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        # feature extraction, patterns going from 64 dim to 1
        self.linear1 = nn.Sequential(nn.Linear(n_embd,1))
        self.SigM1 = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.SigM1(x)

        return x


class Heart_GPT_FineTune(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # mask option in blocks allows you to unmask the last layer if set to False
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer - 1)] + [Block(n_embd, n_head = n_head, mask=True)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# for training

model = Heart_GPT_FineTune()

# load base model
model.load_state_dict(torch.load(model_path))

# freeze base model
for param in model.parameters():
    param.requires_grad = False

#set final linear layer to new linear layer
model.lm_head = NewHead(n_embd)

# make sure new linear layer is trainable
for param in model.lm_head.parameters():
    param.requires_grad = True

# make sure last layer norm is trainable 
for param in model.ln_f.parameters():
    param.requires_grad = True

last_block = model.blocks[-1]  # Get the last block

# make sure all of last block is trainable
for param in last_block.parameters():
    param.requires_grad = True

m = model.to(device)



