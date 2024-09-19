import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QComboBox, QFormLayout
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QIntValidator, QColor
from PyQt5.QtWidgets import QLineEdit, QSlider, QHBoxLayout, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.io import savemat

# Assuming the model and other configurations are already set up
model_config = 'PPG_PT'
block_size = 500
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
# change model directory here to where you have the models stored
model_path_ppg = "D:/HeartGPTModels/PPGPT_500k_iters.pth"
model_path_ppg_AF = "D:/HeartGPTModels/PPGPT_AF_1k_iters.pth"

model_path_ecg = "D:/HeartGPTModels/ECGPT_560k_iters.pth"
model_path_ecg_AF = "D:/HeartGPTModels/ECGPT_AF_1k_iters.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_config == 'PPG_PT':
    vocab_size = 102
    model_path = model_path_ppg
    model_path_AF = model_path_ppg_AF
elif model_config == 'ECG_PT':
    vocab_size = 101
    model_path = model_path_ecg
    model_path_AF = model_path_ecg_AF

# Initialize an empty list to store the weights matrices
weights_matrices = []

# Model definition
class Head(nn.Module):
    def __init__(self, head_size, weights_matrices):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))  # buffer means not updated by optimiser
        self.dropout = nn.Dropout(dropout)
        self.weights_matrices = weights_matrices  # Reference to the external list

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

        self.weights_matrices.append(wei.detach().cpu().numpy())  # Store the weights matrix in the list
        #print(f"weights_matrices size: {np.array(self.weights_matrices).shape}")  # Print the full size of weights_matrices

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, weights_matrices):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, weights_matrices) for _ in range(num_heads)])
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
            nn.Linear(4 * n_embd, n_embd),  # multiplication performed in attention is all you need paper
            # expands and contracts back down to projection
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, weights_matrices):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, weights_matrices)
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
        self.linear1 = nn.Sequential(nn.Linear(n_embd,1),
                                        nn.Dropout(dropout))
        self.Mpool = nn.Sequential(nn.Flatten(),
                                    nn.MaxPool1d(500))
        self.SigM1 = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = x[:,-1,:]
        x = self.SigM1(x)

        return x

class HeartGPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, weights_matrices) for _ in range(n_layer)])
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

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes1 = fig.add_subplot(311)  # First plot (Tokenised Input Data and Generated Output)
        self.axes2 = fig.add_subplot(312)  # Second plot (Difference between third and second plot)
        self.axes3 = fig.add_subplot(313)  # Third plot (Both second and third plots on the same graph)
        super(PlotCanvas, self).__init__(fig)
        fig.tight_layout(pad=3.0)  # Add padding between plots

    def plot(self, input_data=None, output_data=None, context_length=None, attention_weights=None, layer_idx=None, head_idx=None, slider_value=500, attention_weights_AF=None):
        # Initialize final_row and final_row_AF to None
        final_row = None
        final_row_AF = None

        # Plot Tokenised Input Data and Generated Output
        if input_data is not None and output_data is not None and context_length is not None:
            self.axes1.clear()
            self.axes1.plot(input_data.flatten(), color='black', label='Tokenised Input Data')
            if len(output_data) > context_length:
                self.axes1.plot(range(context_length, context_length + len(output_data) - context_length), output_data[context_length:], color='red')
            self.axes1.set_xlabel('Token Number')
            self.axes1.set_ylabel('Token\nValue')  # Use \n to break the ylabel into multiple lines
            self.axes1.legend()

        # Plot Attention Weights for first model
        if attention_weights is not None and layer_idx is not None and head_idx is not None:
            index = layer_idx * 8 + head_idx
            if index < len(attention_weights):
                attention_weights = attention_weights[index][0]
                final_row = attention_weights[slider_value - 1]  # Adjust for 0-based index

        # Plot Attention Weights for second model
        if attention_weights_AF is not None and layer_idx is not None and head_idx is not None:
            index = layer_idx * 8 + head_idx
            if index < len(attention_weights_AF):
                attention_weights_AF = attention_weights_AF[index][0]
                final_row_AF = attention_weights_AF[slider_value - 1]  # Adjust for 0-based index

        # Plot the difference between the third and second plot
        if final_row is not None and final_row_AF is not None:
            difference_row = final_row_AF - final_row
            self.axes2.clear()
            self.axes2.plot(difference_row, color='red', label=f'Difference: Layer {layer_idx + 1}, Head {head_idx + 1}, Token {slider_value}')
            self.axes2.set_xlabel('Token Number')
            self.axes2.set_ylabel('Attention\nWeight\nDifference')  # Use \n to break the ylabel into multiple lines
            self.axes2.legend()
        else:
            self.axes2.clear()
            self.axes2.text(0.5, 0.5, 'Attention weights loading...', horizontalalignment='center', verticalalignment='center', transform=self.axes2.transAxes)

        # Plot both the second and third plots on the same graph
        if final_row is not None and final_row_AF is not None:
            self.axes3.clear()
            self.axes3.plot(final_row, color='black', label=f'First Model: Layer {layer_idx + 1}, Head {head_idx + 1}, Token {slider_value}')
            self.axes3.plot(final_row_AF, color='blue', label=f'Second Model: Layer {layer_idx + 1}, Head {head_idx + 1}, Token {slider_value}')
            self.axes3.set_xlabel('Token Number')
            self.axes3.set_ylabel('Attention\nWeight')  # Use \n to break the ylabel into multiple lines
            self.axes3.legend()
        else:
            self.axes3.clear()
            self.axes3.text(0.5, 0.5, 'Attention weights loading...', horizontalalignment='center', verticalalignment='center', transform=self.axes3.transAxes)

        # Draw the plots
        self.draw()








class Worker(QThread):
    finished_signal = pyqtSignal(list)

    def __init__(self, model, example_context_tensor, max_new_tokens):
        super().__init__()
        self.model = model
        self.example_context_tensor = example_context_tensor
        self.max_new_tokens = max_new_tokens

    def run(self):
        output = self.model.generate(self.example_context_tensor, max_new_tokens=self.max_new_tokens)[0].tolist()
        self.finished_signal.emit(output)

class App(QMainWindow):
    update_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.title = 'HeartGPT: AFib Interpretability'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.model_config = 'PPG_PT'
        self.initUI()
        self.first_model_done = False
        self.second_model_done = False

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
        self.model_selector.currentIndexChanged.connect(self.update_model_config)

        # Remove the layer and head selectors
        # self.layer_selector = QComboBox(self)
        # for i in range(1, 9):
        #     self.layer_selector.addItem(str(i))
        # form_layout.addRow('Select Layer:', self.layer_selector)

        # self.head_selector = QComboBox(self)
        # for i in range(1, 9):
        #     self.head_selector.addItem(str(i))
        # self.head_selector.addItem("All")  # Add the "All" option
        # form_layout.addRow('Select Head:', self.head_selector)

        # self.layer_selector.currentIndexChanged.connect(self.update_attention_plot)
        # self.head_selector.currentIndexChanged.connect(self.update_attention_plot)

        self.layout.addLayout(form_layout)

        self.button = QPushButton('Load Context and Estimate AFib', self)  # Updated button text
        self.button.clicked.connect(self.load_and_plot_data)
        self.layout.addWidget(self.button)

        self.save_button = QPushButton('Save Attention Weights', self)  # Updated button text
        self.save_button.clicked.connect(self.save_attention_weights)
        self.layout.addWidget(self.save_button)

        self.progress_label = QLabel('Ready', self)
        self.layout.addWidget(self.progress_label)

        # Add a checkbox to toggle y-axis limit setting
        #self.y_axis_checkbox = QCheckBox('Fix Y-Axis Limit', self)
        #self.y_axis_checkbox.setChecked(True)  # Default to checked
        #self.y_axis_checkbox.stateChanged.connect(self.update_attention_plot)
        #self.layout.addWidget(self.y_axis_checkbox)

        self.probability_label = QLabel('Estimated Probability of Atrial Fibrillation: ', self)
        self.layout.addWidget(self.probability_label)


    def update_model_config(self):
        selected_model = self.model_selector.currentText()
        if selected_model == "PPGPT":
            self.model_config = 'PPG_PT'
        elif selected_model == "ECGPT":
            self.model_config = 'ECG_PT'
        self.update_model_parameters()

    def update_model_parameters(self):
        global vocab_size, model_path, model_path_AF
        if self.model_config == 'PPG_PT':
            vocab_size = 102
            model_path = model_path_ppg
            model_path_AF = model_path_ppg_AF
        elif self.model_config == 'ECG_PT':
            vocab_size = 101
            model_path = model_path_ecg
            model_path_AF = model_path_ecg_AF

    def save_attention_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder:
            output_file = f"{folder}/Attention_weights.mat"
            # Save the full 64x1x500x500 weights in a .mat file
            savemat(output_file, {'attention_weights': np.array(weights_matrices)})
            self.progress_label.setText(f'Saved to {output_file}')


    def update_attention_plot(self):
        # Set the default layer and head indices
        layer_idx = 7  # 8th layer (0-based index)
        head_idx = "All"  # "All" heads
        slider_value = 500  # Fixed slider value

        if weights_matrices and self.first_model_done and self.second_model_done:
            if head_idx == "All":
                # Sum the weights from all heads for the selected layer
                attention_weights = sum(weights_matrices[layer_idx * 8 + i][0] for i in range(8))
                attention_weights_AF = sum(weights_matrices[64 + layer_idx * 8 + i][0] for i in range(8))
            else:
                attention_weights = weights_matrices[layer_idx * 8 + head_idx][0]
                attention_weights_AF = weights_matrices[64 + layer_idx * 8 + head_idx][0]

            final_row = attention_weights[slider_value - 1]  # Adjust for 0-based index
            final_row_AF = attention_weights_AF[slider_value - 1]  # Adjust for 0-based index

            # Plot the difference between the third and second plot
            difference_row = final_row_AF - final_row
            self.plot_canvas.axes2.clear()
            self.plot_canvas.axes2.plot(difference_row, color='red', label=f'Attention Change: Final Layer')
            self.plot_canvas.axes2.set_xlabel('Token Number')
            self.plot_canvas.axes2.set_ylabel('Attention\nWeight\nDifference')
            self.plot_canvas.axes2.legend()

            # Plot both the second and third plots on the same graph
            self.plot_canvas.axes3.clear()
            self.plot_canvas.axes3.plot(final_row, color='black', label=f'Pre-trained Model: Final Layer')
            self.plot_canvas.axes3.plot(final_row_AF, color='blue', label=f'AF Fine-tuned Model: Final Layer')
            self.plot_canvas.axes3.set_xlabel('Token Number')
            self.plot_canvas.axes3.set_ylabel('Attention\nWeight')
            self.plot_canvas.axes3.legend()

            # Always scale the y-axis limit between token numbers 25 and 500
            max_value = np.max(attention_weights[slider_value - 1, 24:])
            max_value_AF = np.max(attention_weights_AF[slider_value - 1, 24:])
            max_value_diff = np.max(difference_row[24:])
            if len(final_row) > 24:
                self.plot_canvas.axes2.set_ylim(0, max_value_diff)
                self.plot_canvas.axes3.set_ylim(0, max_value_AF)
            else:
                self.plot_canvas.axes2.set_ylim(0, np.max(final_row))
                self.plot_canvas.axes3.set_ylim(0, np.max(final_row_AF))

            self.plot_canvas.axes2.legend()
            self.plot_canvas.axes3.legend()
            # Draw the updated plot
            self.plot_canvas.draw()




    def update_slider_plot(self, value=500):
        # Set the default layer and head indices
        layer_idx = 7  # 8th layer (0-based index)
        head_idx = "All"  # "All" heads

        if weights_matrices and self.first_model_done and self.second_model_done:
            if head_idx == "All":
                # Sum the weights from all heads for the selected layer
                attention_weights = sum(weights_matrices[layer_idx * 8 + i][0] for i in range(8))
                attention_weights_AF = sum(weights_matrices[64 + layer_idx * 8 + i][0] for i in range(8))
            else:
                attention_weights = weights_matrices[layer_idx * 8 + head_idx][0]
                attention_weights_AF = weights_matrices[64 + layer_idx * 8 + head_idx][0]

            final_row = attention_weights[value - 1]  # Adjust for 0-based index
            final_row_AF = attention_weights_AF[value - 1]  # Adjust for 0-based index

            # Plot the difference between the third and second plot
            difference_row = final_row_AF - final_row
            self.plot_canvas.axes2.clear()
            self.plot_canvas.axes2.plot(difference_row, color='red', label=f'Attention Change: Final Layer')
            self.plot_canvas.axes2.set_xlabel('Token Number')
            self.plot_canvas.axes2.set_ylabel('Attention\nWeight\nDifference')
            self.plot_canvas.axes2.legend()

            # Plot both the second and third plots on the same graph
            self.plot_canvas.axes3.clear()
            self.plot_canvas.axes3.plot(final_row, color='black', label=f'Pre-trained Model: Final Layer')
            self.plot_canvas.axes3.plot(final_row_AF, color='blue', label=f'AF Fine-tuned Model: Final Layer')
            self.plot_canvas.axes3.set_xlabel('Token Number')
            self.plot_canvas.axes3.set_ylabel('Attention\nWeight')
            self.plot_canvas.axes3.legend()

            # Always scale the y-axis limit between token numbers 25 and 500
            max_value = np.max(attention_weights[value - 1, 24:])
            max_value_AF = np.max(attention_weights_AF[value - 1, 24:])
            max_value_diff = np.max(difference_row[24:])
            if len(final_row) > 24:
                self.plot_canvas.axes2.set_ylim(0, max_value_diff)
                self.plot_canvas.axes3.set_ylim(0, max_value_AF)
            else:
                self.plot_canvas.axes2.set_ylim(0, np.max(final_row))
                self.plot_canvas.axes3.set_ylim(0, np.max(final_row_AF))

            self.plot_canvas.axes2.legend()
            self.plot_canvas.axes3.legend()

            # Update the top plot with a blue circle at the position of the token number chosen by the slider
            self.plot_canvas.axes1.clear()
            input_data = self.worker.example_context_tensor.cpu().numpy().flatten()
            context_length = self.worker.example_context_tensor.shape[1]
            self.plot_canvas.axes1.plot(input_data, color='black', label='Tokenised Input Data')  # Plot the blue circle
            self.plot_canvas.axes1.set_xlabel('Token Number')
            self.plot_canvas.axes1.set_ylabel('Token\nValue')
            self.plot_canvas.axes1.legend()

            # Draw the updated plots
            self.plot_canvas.draw()




    def update_slider_label(self, value):
        # Update the label with the current slider value
        self.slider_label.setText(str(value))



    def save_generated_tokens(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder:
            output_file = f"{folder}/outputs.csv"
            np.savetxt(output_file, [self.generated_token], delimiter=",")
            self.progress_label.setText(f'Saved to {output_file}')

    def load_and_plot_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        input_file, _ = QFileDialog.getOpenFileName(self, "Load Input CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if input_file:
            # Reset the weights_matrices list
            global weights_matrices
            weights_matrices = []

            # Load the CSV file into a DataFrame
            df = pd.read_csv(input_file, header=None)
            # Convert the DataFrame to a numpy array
            input_data = df.values

            selected_model = self.model_selector.currentText()
            if selected_model == "PPGPT":
                self.model_config = 'PPG_PT'
                plot_title = "PPGPT Generation"
            elif selected_model == "ECGPT":
                self.model_config = 'ECG_PT'
                plot_title = "ECGPT Generation"
            self.update_model_parameters()

            class HeartGPTModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
                    self.position_embedding_table = nn.Embedding(block_size, n_embd)
                    self.blocks = nn.Sequential(*[Block(n_embd, n_head, weights_matrices) for _ in range(n_layer)])
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

                def generate(self, idx, max_new_tokens):
                    for i in range(max_new_tokens):
                        idx_cond = idx[:, -block_size:]
                        logits, loss = self(idx_cond)
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)
                    return idx

            model = HeartGPTModel()
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            model.to(device)

            self.model_AF = HeartGPTModel()
            self.model_AF.lm_head = NewHead(n_embd)
            self.model_AF.load_state_dict(torch.load(model_path_AF, weights_only=True))
            self.model_AF.to(device)
            self.model_AF.eval()

            data_tokenised = tokenize_biosignal(input_data)
            example_context_tensor = torch.tensor(data_tokenised, dtype=torch.long, device=device)

            self.plot_canvas.axes1.set_title(plot_title)
            self.plot_canvas.plot(input_data=example_context_tensor.cpu().numpy().flatten(), output_data=[], context_length=example_context_tensor.shape[1])

            self.worker = Worker(model, example_context_tensor, max_new_tokens=1)
            self.worker.finished_signal.connect(self.plot_output)
            self.worker.start()

            self.save_button.setEnabled(False)



    def run_second_model(self, model_AF, example_context_tensor):
        global weights_matrices_AF
        weights_matrices_AF = []
        with torch.no_grad():
            probability_tensor, _ = model_AF(example_context_tensor)  # Get the tensor output
            probability = probability_tensor.cpu().numpy()[0][0]  # Convert to NumPy array and extract the value
        self.progress_label.setText('AF model run complete')
        self.save_button.setEnabled(True)

        # Set the flag for the second model
        self.second_model_done = True

        # Update the probability label with conditional color
        if probability >= 0.5:
            color = 'red'
        else:
            color = 'green'
        self.probability_label.setStyleSheet(f'color: {color}')
        self.probability_label.setText(f'Estimated Probability of Atrial Fibrillation: {probability:.2f}')

        # Update the plots only if both models are done
        if self.first_model_done and self.second_model_done:
            self.update_slider_plot(500)  # Fixed slider value





    def plot_output(self, output):
        self.generated_tokens = output
        self.plot_canvas.plot(input_data=self.worker.example_context_tensor.cpu().numpy().flatten(), output_data=output, context_length=self.worker.example_context_tensor.shape[1])
        self.progress_label.setText('Generation complete')
        self.save_button.setEnabled(True)

        # Plot the blue circle at the position of the token number chosen by the slider
        

        # Set the flag for the first model
        self.first_model_done = True

        # Run the second model
        self.run_second_model(self.model_AF, self.worker.example_context_tensor)









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


