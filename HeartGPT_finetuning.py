import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

# Harry Davies 19_09_2024

# The following code is adapted from a tutorial by Andrej Kapathy, available at https://github.com/karpathy/ng-video-lecture
# The explaination behind this code and the model files can be found in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data"
# available at https://arxiv.org/abs/2407.20775


model_config = 'ECG_PT' #switch between 'ECG_PT' and 'PPG_PT'

eval_interval = 50
save_interval = 20000
max_iters = 5000
eval_iters = 50
batch_size = 128 # sequences we process in parellel
block_size = 500 # this is context length
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
learning_rate = 3e-04
model_path_ppg = "D:/HeartGPTModels/PPGPT_500k_iters.pth"
model_path_ecg = "D:/HeartGPTModels/ECGPT_560k_iters.pth"

model_path_finetune = "D:/HeartGPTModels/HeartGPT_finetune_example.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_config == 'PPG_PT':
    vocab_size = 102 #102 for PPGPT, 101 for ECGPT
    model_path = model_path_ppg
elif model_config == 'ECG_PT':
    vocab_size = 101
    model_path = model_path_ecg

# load in the data, in our case data was originally prepared in matlab
# original fine tuning data for AFib had training data (X) dimensions of Nx500, and label dimensions (Y) of Nx2. 
# For afib pne of the labels in Y was subject number, and used to exclude subjects during cross-validation. The first of the 2 values was the AF class of 0 or 1.
# For beat detection fine tuning, training data (X) had dimensions of Nx500, and label dimenions (Y) of Nx500, where 0 corresponded to no beat, and 1 was labelled at the position of a beat.
data_load = scipy.io.loadmat('D:/training_data.mat')
X = data_load['rounded_output_store']
y = data_load['label_store']




# Get the permutation of indices
perm = np.random.permutation(X.shape[0])

# Shuffle X and y
X_shuffled = X[perm]
y_shuffled = y[perm]

# split into train and test
trainX, testX, trainy, testy = train_test_split(X_shuffled, y_shuffled, test_size=0.1, random_state=10)

def get_batch_AF(split):
    dataX = trainX  
    datay = trainy  
    ix = torch.randint(len(dataX), (batch_size,))
    x = torch.stack([dataX[i,:] for i in ix])
    y = torch.stack([datay[i,0] for i in ix])
    y = y.clamp(0,1)
    #for AFib, label y is one value of either 0 or 1
  
    x, y = x.to(device), y.to(device)
    return x, y

def get_batch_beat(split):
    dataX = trainX  if split == 'train' else testX
    datay = trainy  if split == 'train' else testy
    ix = torch.randint(len(dataX), (batch_size,))
    x = torch.stack([dataX[i,:] for i in ix])
    y = torch.stack([datay[i,:] for i in ix])
    # labels in this case are same dimension as input, but still between 0 and 1    
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss_AF():
    out_loss = {}
    out_acc = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accuracy_store = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_AF(split)
            logits = model(X, Y)
            Y = Y.float()
            logits_reshaped = logits.reshape(-1)
            logits_reshaped = logits_reshaped.clamp(0,1)
            loss = criterion(logits_reshaped,Y)

            logits_reshaped_val = (logits_reshaped > 0.5).float()
            accuracy = (logits_reshaped_val == Y).float().mean()
            accuracy_store[k] = accuracy.item()
            losses[k] = loss.item()
        out_loss[split] = losses.mean()
        out_acc[split] = accuracy_store.mean()

    model.train()
    return out_loss, out_acc

@torch.no_grad()
def estimate_loss_beat():
    out_loss = {}
    out_acc = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accuracy_store = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_beat(split)
            logits = model(X, Y)
            Y = Y.float()
            logits_reshaped = logits.reshape(-1)
            y_reshaped = Y.reshape(-1)
            loss = criterion(logits_reshaped,y_reshaped)

                        # Calculate true positives
            logits_reshaped_val = (logits_reshaped > 0.5).float()
            y_reshaped_val = (y_reshaped > 0.5).float()
            true_positives = (logits_reshaped_val * y_reshaped_val).sum()

            # Calculate the total number of positive predictions
            total_positives = logits_reshaped_val.sum()

            # Calculate the percentage of true positives
            percentage_true_positives = true_positives / total_positives if total_positives != 0 else 0


            accuracy_store[k] = percentage_true_positives.item()
            losses[k] = loss.item()
        out_loss[split] = losses.mean()
        out_acc[split] = accuracy_store.mean()
    model.train()
    return out_loss, out_acc


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
        #x1 = x1[:,-1,:]   #for classification problems (e.g AFib) you need just the last value, for beat detection you need all 500.
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

criterion = nn.BCELoss()

optimizer  = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    #if you want to evaluate loss throughout fine tuning
    if iter % eval_interval == 0:
        losses, accuracies = estimate_loss_beat()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"step {iter}: train accuracy {accuracies['train']:.4f}, val accuracy {accuracies['val']:.4f}")

    xb, yb = get_batch_beat('train')
    logits = m(xb, yb)
    yb = yb.float()
    logits_reshaped = logits.reshape(-1)
    #logits_reshaped = logits_reshaped.clamp(0,1) clamping could be required
    yb_reshaped = yb.reshape(-1)
    loss = criterion(logits_reshaped,yb_reshaped)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(), model_path_finetune)



