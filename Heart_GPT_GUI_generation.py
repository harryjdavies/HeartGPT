import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import QLineEdit, QFormLayout
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtGui import QIntValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Assuming the model and other configurations are already set up
model_config = 'PPG_PT'
block_size = 500
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
# change model directory here to where models are stored
model_path_ppg = "D:/HeartGPTModels/PPGPT_500k_iters.pth"
model_path_ecg = "D:/HeartGPTModels/ECGPT_560k_iters.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_config == 'PPG_PT':
    vocab_size = 102
    model_path = model_path_ppg
elif model_config == 'ECG_PT':
    vocab_size = 101
    model_path = model_path_ecg

# Model definition
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))  # buffer means not updated by optimiser
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention weights
        wei = q @ k.transpose(-2, -1) * C**-0.5  # square root headsize # (B, T, C) @ (B, C, T) = B, T, T
        # for every batch, we will now have a T by T matrix giving us the affinities of each token
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # the tril signifies a decoder block, future tokens cannot communicate with the past
        wei = F.softmax(wei, dim=-1)  # weights corresponding to the update of each token sum to 1

        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # creating a list of head objects (turned into modules) resulting in a number of head modules
        # then assigns the list of modules to self.heads - these run in parallel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # projection generally matches sizes for adding in residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate the output of the different attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # multiplication performed in attention is all you need paper
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


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot(self, input_data, output_data, context_length):
        self.axes.clear()
        self.axes.plot(input_data, color='black', label='Tokenised Input Data')
        if len(output_data) > context_length:
            self.axes.plot(range(context_length, context_length + len(output_data) - context_length), output_data[context_length:], color='red', label='Generated Output')
        self.axes.set_xlabel('Token Number')
        self.axes.set_ylabel('Token Value')
        self.axes.legend()
        self.draw()



class Worker(QThread):
    update_signal = pyqtSignal(int, int, torch.Tensor)
    finished_signal = pyqtSignal(list)

    def __init__(self, model, example_context_tensor, max_new_tokens, update_interval):
        super().__init__()
        self.model = model
        self.example_context_tensor = example_context_tensor
        self.max_new_tokens = max_new_tokens
        self.update_interval = update_interval

    def run(self):
        def update_progress(current, total, idx):
            self.update_signal.emit(current, total, idx)

        output = self.model.generate(self.example_context_tensor, max_new_tokens=self.max_new_tokens, callback=update_progress, update_interval=self.update_interval)[0].tolist()
        self.finished_signal.emit(output)


class App(QMainWindow):
    update_signal = pyqtSignal(int, int, torch.Tensor)

    def __init__(self):
        super().__init__()
        self.title = 'HeartGPT Model GUI'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.plot_canvas = PlotCanvas(self)
        self.layout.addWidget(self.plot_canvas)

        form_layout = QFormLayout()

        self.model_selector = QComboBox(self)
        self.model_selector.addItem("PPGPT")
        self.model_selector.addItem("ECGPT")
        form_layout.addRow('Select Model:', self.model_selector)

        self.max_tokens_input = QLineEdit(self)
        self.max_tokens_input.setText('500')
        self.max_tokens_input.setValidator(QIntValidator())
        form_layout.addRow('Max Tokens:', self.max_tokens_input)

        self.update_interval_input = QLineEdit(self)
        self.update_interval_input.setText('2')
        self.update_interval_input.setValidator(QIntValidator())
        form_layout.addRow('Update Interval:', self.update_interval_input)

        self.layout.addLayout(form_layout)

        self.button = QPushButton('Load Context and Generate', self)
        self.button.clicked.connect(self.load_and_plot_data)
        self.layout.addWidget(self.button)

        self.save_button = QPushButton('Save Generated Tokens', self)
        self.save_button.clicked.connect(self.save_generated_tokens)
        self.layout.addWidget(self.save_button)

        self.progress_label = QLabel('Ready', self)
        self.layout.addWidget(self.progress_label)


    def save_generated_tokens(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder:
            output_file = f"{folder}/outputs.csv"
            np.savetxt(output_file, self.generated_tokens, delimiter=",")
            self.progress_label.setText(f'Saved to {output_file}')



    def load_and_plot_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        input_file, _ = QFileDialog.getOpenFileName(self, "Load Input CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if input_file:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(input_file, header=None)
            # Convert the DataFrame to a numpy array
            input_data = df.values

            selected_model = self.model_selector.currentText()
            if selected_model == "PPGPT":
                model_path = model_path_ppg
                plot_title = "PPGPT Generation"
                vocab_size = 102
            else:
                model_path = model_path_ecg
                plot_title = "ECGPT Generation"
                vocab_size = 101

            class HeartGPTModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
                    self.position_embedding_table = nn.Embedding(block_size, n_embd)
                    self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
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
                    if targets is None:
                        loss = None
                    else:
                        B, T, C = logits.shape
                        logits = logits.view(B * T, C)
                        targets = targets.view(B * T)
                        loss = F.cross_entropy(logits, targets)
                    return logits, loss

                def generate(self, idx, max_new_tokens, callback=None, update_interval=10):
                    for i in range(max_new_tokens):
                        idx_cond = idx[:, -block_size:]
                        logits, loss = self(idx_cond)
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)
                        if callback and (i + 1) % update_interval == 0:
                            callback(i + 1, max_new_tokens, idx)
                    return idx

            model = HeartGPTModel()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to(device)

            data_tokenised = tokenize_biosignal(input_data)
            example_context_tensor = torch.tensor(data_tokenised, dtype=torch.long, device=device)

            self.plot_canvas.axes.set_title(plot_title)
            self.plot_canvas.plot(example_context_tensor.cpu().numpy().flatten(), [], example_context_tensor.shape[1])

            max_tokens = int(self.max_tokens_input.text())
            update_interval = int(self.update_interval_input.text())

            self.worker = Worker(model, example_context_tensor, max_new_tokens=max_tokens, update_interval=update_interval)
            self.worker.update_signal.connect(self.update_progress)
            self.worker.finished_signal.connect(self.plot_output)
            self.worker.start()

            self.save_button.setEnabled(False)






    def update_progress(self, current, total, idx):
        self.progress_label.setText(f'Generating {current}/{total} new tokens')
        self.plot_canvas.plot(self.worker.example_context_tensor.cpu().numpy().flatten(), idx.cpu().numpy().flatten(), self.worker.example_context_tensor.shape[1])

    def plot_output(self, output):
        self.generated_tokens = output
        self.plot_canvas.plot(self.worker.example_context_tensor.cpu().numpy().flatten(), output, self.worker.example_context_tensor.shape[1])
        self.progress_label.setText('Generation complete')
        self.save_button.setEnabled(True)



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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
