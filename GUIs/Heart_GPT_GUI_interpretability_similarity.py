import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QComboBox, QFormLayout
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QLineEdit, QSlider, QHBoxLayout, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.io import savemat
from sklearn.metrics.pairwise import cosine_similarity

# Assuming the model and other configurations are already set up
model_config = 'PPG_PT'
block_size = 500
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
# change model directory here to where you have the models stored
model_path_ppg = "D:/HeartGPTModels/PPGPT_500k_iters.pth"
model_path_ecg = "D:/HeartGPTModels/ECGPT_560k_iters.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_config == 'PPG_PT':
    vocab_size = 102
    model_path = model_path_ppg
elif model_config == 'ECG_PT':
    vocab_size = 101
    model_path = model_path_ecg

# Initialize an empty list to store the weights matrices
weights_matrices = []
blocks_list = []

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

        #self.weights_matrices.append(wei.detach().cpu().numpy())  # Store the weights matrix in the list
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
        self.blocks_list = blocks_list# Initialize the list to store the blocks

    def forward(self, x):
        x_input_block = x.detach().cpu().numpy()
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x_output_block = x.detach().cpu().numpy()
        #print(f"output block size: {np.array(x_output_block).shape}")
        # Append the blocks to the list
        if len(self.blocks_list) == 0:
            self.blocks_list.append(np.array(x_input_block))
        self.blocks_list.append(np.array(x_output_block))
        #print(f"block list size: {np.array(self.blocks_list).shape}")

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
        self.axes1 = fig.add_subplot(211)  # First plot (Tokenised Input Data and Generated Output)
        self.axes2 = fig.add_subplot(212)  # Second plot (Cosine Similarity Sequences)
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot(self, input_data=None, output_data=None, context_length=None, cos_sim_sequence1=None, cos_sim_sequence2=None, cos_sim_sequence3=None):
        # Plot Tokenised Input Data and Generated Output
        if input_data is not None and output_data is not None and context_length is not None:
            self.axes1.clear()
            self.axes1.plot(input_data.flatten(), color='black', label='Tokenised Input Data')
            if len(output_data) > context_length:
                self.axes1.plot(range(context_length, context_length + len(output_data) - context_length), output_data[context_length:], color='red')
            self.axes1.set_xlabel('Token Number')
            self.axes1.set_ylabel('Token Value')
            self.axes1.legend()

        # Plot Cosine Similarity Sequences
        if cos_sim_sequence1 is not None and cos_sim_sequence2 is not None and cos_sim_sequence3 is not None:
            self.axes2.clear()
            self.axes2.plot(cos_sim_sequence1, color='blue', label='Cosine Similarity (Blue & Blue)')
            self.axes2.plot(cos_sim_sequence2, color='red', label='Cosine Similarity (Red & Blue)')
            self.axes2.plot(cos_sim_sequence3, color='green', label='Cosine Similarity (Green & Blue)')
            self.axes2.set_ylabel('Cosine Similarity')
            self.axes2.set_ylim(-0.4, 1.01)
            self.axes2.legend()
            x_labels = ['Input'] + [f'Block {i}\nOutput' for i in range(1, 9)]
            self.axes2.set_xticks(range(9))
            self.axes2.set_xticklabels(x_labels, fontsize=8)

        # Draw the plots
        self.draw()




class Worker(QThread):
    finished_signal = pyqtSignal(list)
    blocks_list_signal = pyqtSignal(list)

    def __init__(self, model, example_context_tensor, max_new_tokens):
        super().__init__()
        self.model = model
        self.example_context_tensor = example_context_tensor
        self.max_new_tokens = max_new_tokens

    def run(self):
        output = self.model.generate(self.example_context_tensor, max_new_tokens=self.max_new_tokens)[0].tolist()
        #blocks_list = self.model.get_blocks_list()  # Retrieve the blocks_list
        self.finished_signal.emit(output)
        #self.blocks_list_signal.emit(blocks_list)  # Emit the blocks_list

class App(QMainWindow):
    update_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.title = 'Heart GPT Interpretability: Vector Similarity'
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

        # Add the information button above the plots
        info_layout = QHBoxLayout()
        self.info_button = QPushButton('Information', self)
        self.info_button.setToolTip('This GUI is designed to help users interpret the operation of the HeartGPT models.\nFor each token in the input context, you can compare the vector similarity with other tokens as they propagate through the transformer layers.\nThere is a relevant example of this with rising slopes and falling slopes in the paper "Interpretable Pre-Trained Transformers for Heart Time-Series Data".')
        self.info_button.setFixedSize(120, 20)  # Make the button smaller
        info_layout.addWidget(self.info_button)
        info_layout.addStretch()  # Add stretch to push the button to the left
        self.layout.addLayout(info_layout)

        self.plot_canvas = PlotCanvas(self)
        self.layout.addWidget(self.plot_canvas)

        form_layout = QFormLayout()

        self.model_selector = QComboBox(self)
        self.model_selector.addItem("PPGPT")
        self.model_selector.addItem("ECGPT")
        form_layout.addRow('Select Model:', self.model_selector)

        # Add the slider and label to a horizontal layout
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(1, 500)
        self.slider.setValue(500)  # Set default value to 500
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(50)

        # Calculate the width of the plot area and set the slider width accordingly
        plot_width = self.plot_canvas.size().width()
        self.slider.setFixedWidth(plot_width - 50)  # Adjust the width of the slider to match the plot width minus 10mm

        # Add a left margin to shift the slider to the right by 5mm
        slider_layout.setContentsMargins(30, 0, 0, 0)
        slider_layout.addWidget(self.slider)

        self.slider_label = QLabel('500', self)  # Set default label to 500
        self.slider_label.setFixedWidth(50)  # Adjust the width of the label
        slider_layout.addWidget(self.slider_label)

        form_layout.addRow('Select Comparison Token:', slider_layout)

        self.slider.valueChanged.connect(self.update_slider_plot)
        self.slider.valueChanged.connect(self.update_slider_label)

        self.layout.addLayout(form_layout)

        # Add the second slider and label to a horizontal layout
        slider_layout2 = QHBoxLayout()
        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider2.setRange(1, 500)
        self.slider2.setValue(500)  # Set default value to 500
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(50)

        self.slider2.setFixedWidth(plot_width - 50)  # Adjust the width of the slider to match the plot width minus 10mm
        slider_layout2.setContentsMargins(30, 0, 0, 0)
        slider_layout2.addWidget(self.slider2)

        self.slider_label2 = QLabel('500', self)  # Set default label to 500
        self.slider_label2.setFixedWidth(50)  # Adjust the width of the label
        slider_layout2.addWidget(self.slider_label2)

        form_layout.addRow('Select Token (Red):', slider_layout2)

        self.slider2.valueChanged.connect(self.update_slider_plot)
        self.slider2.valueChanged.connect(self.update_slider_label2)

        # Add the third slider and label to a horizontal layout
        slider_layout3 = QHBoxLayout()
        self.slider3 = QSlider(Qt.Horizontal, self)
        self.slider3.setRange(1, 500)
        self.slider3.setValue(500)  # Set default value to 500
        self.slider3.setTickPosition(QSlider.TicksBelow)
        self.slider3.setTickInterval(50)

        self.slider3.setFixedWidth(plot_width - 50)  # Adjust the width of the slider to match the plot width minus 10mm
        slider_layout3.setContentsMargins(30, 0, 0, 0)
        slider_layout3.addWidget(self.slider3)

        self.slider_label3 = QLabel('500', self)  # Set default label to 500
        self.slider_label3.setFixedWidth(50)  # Adjust the width of the label
        slider_layout3.addWidget(self.slider_label3)

        form_layout.addRow('Select Token (Green):', slider_layout3)

        self.slider3.valueChanged.connect(self.update_slider_plot)
        self.slider3.valueChanged.connect(self.update_slider_label3)

        self.layout.addLayout(form_layout)

        self.button = QPushButton('Load Context and Initialise', self)  # Updated button text
        self.button.clicked.connect(self.load_and_plot_data)
        self.layout.addWidget(self.button)

        self.save_button = QPushButton('Save Vector Embedding of Tokens', self)  # Updated button text
        self.save_button.clicked.connect(self.save_blocks_list)
        self.layout.addWidget(self.save_button)

        self.progress_label = QLabel('Ready', self)
        self.layout.addWidget(self.progress_label)


    def save_blocks_list(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder:
            output_file = f"{folder}/vector_embedding.mat"
            # Save the blocks_list in a .mat file
            savemat(output_file, {'vector_propagation': np.array(blocks_list)})
            self.progress_label.setText(f'Saved to {output_file}')



    def update_slider_plot(self, value):
        slider_pos1 = self.slider.value() - 1
        slider_pos2 = self.slider2.value() - 1
        slider_pos3 = self.slider3.value() - 1

        # Convert each block to a numpy array before slicing
        sequence1 = np.array(blocks_list)[:,:,slider_pos1:slider_pos1+1,:]
        sequence2 = np.array(blocks_list)[:,:,slider_pos2:slider_pos2+1,:]
        sequence3 = np.array(blocks_list)[:,:,slider_pos3:slider_pos3+1,:]

        # Convert to numpy arrays for easier handling
        sequence1 = np.array(sequence1)
        sequence2 = np.array(sequence2)
        sequence3 = np.array(sequence3)


        # Calculate cosine similarity
        # Calculate cosine similarity for each 1x1x1x64 slice
        cos_sim_sequence1 = np.zeros((9,1))
        cos_sim_sequence2 = np.zeros((9,1))
        cos_sim_sequence3 = np.zeros((9,1))
        for x in range(9):
            cos_sim_sequence1[x] = cosine_similarity(sequence1[x,:,:,:].reshape(1,64),sequence1[x,:,:,:].reshape(1,64))
            cos_sim_sequence2[x] = cosine_similarity(sequence1[x,:,:,:].reshape(1,64),sequence2[x,:,:,:].reshape(1,64))
            cos_sim_sequence3[x] = cosine_similarity(sequence1[x,:,:,:].reshape(1,64),sequence3[x,:,:,:].reshape(1,64))

        cos_sim_sequence1 = np.array(cos_sim_sequence1)
        cos_sim_sequence2 = np.array(cos_sim_sequence2)
        cos_sim_sequence3 = np.array(cos_sim_sequence3)


        # Update the plot with the new cosine similarity sequences
        self.plot_canvas.plot(cos_sim_sequence1=cos_sim_sequence1, cos_sim_sequence2=cos_sim_sequence2, cos_sim_sequence3=cos_sim_sequence3)

        # Update the top plot with circles at the positions of the token numbers chosen by the sliders
        self.plot_canvas.axes1.clear()
        input_data = self.worker.example_context_tensor.cpu().numpy().flatten()
        context_length = self.worker.example_context_tensor.shape[1]
        self.plot_canvas.axes1.plot(input_data, color='black', label='Tokenised Input Data')
        self.plot_canvas.axes1.scatter(self.slider.value() - 1, input_data[self.slider.value() - 1], color='blue', s=50)  # Plot the blue circle
        self.plot_canvas.axes1.scatter(self.slider2.value() - 1, input_data[self.slider2.value() - 1], color='red', s=50)  # Plot the red circle
        self.plot_canvas.axes1.scatter(self.slider3.value() - 1, input_data[self.slider3.value() - 1], color='green', s=50)  # Plot the green circle
        self.plot_canvas.axes1.set_xlabel('Token Number')
        self.plot_canvas.axes1.set_ylabel('Token Value')
        self.plot_canvas.axes1.legend()

        self.plot_canvas.draw()




    def update_slider_label(self, value):
        # Update the label with the current slider value
        self.slider_label.setText(str(value))

    def update_slider_label2(self, value):
        self.slider_label2.setText(str(value))

    def update_slider_label3(self, value):
        self.slider_label3.setText(str(value))




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
            global blocks_list
            blocks_list = []

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

            data_tokenised = tokenize_biosignal(input_data)
            example_context_tensor = torch.tensor(data_tokenised, dtype=torch.long, device=device)

            self.plot_canvas.axes1.set_title(plot_title)
            self.plot_canvas.plot(input_data=example_context_tensor.cpu().numpy().flatten(), output_data=[], context_length=example_context_tensor.shape[1])

            self.worker = Worker(model, example_context_tensor, max_new_tokens=1)
            self.worker.finished_signal.connect(self.plot_output)
            #self.worker.blocks_list_signal.connect(self.handle_blocks_list)
            self.worker.start()

            self.save_button.setEnabled(False)






    def plot_output(self, output):
        self.generated_tokens = output
        self.plot_canvas.plot(input_data=self.worker.example_context_tensor.cpu().numpy().flatten(), output_data=output, context_length=self.worker.example_context_tensor.shape[1])
        self.progress_label.setText('Generation complete')
        self.save_button.setEnabled(True)

        # Plot the blue circle at the position of the token number chosen by the slider
        slider_value = self.slider.value()
        self.plot_canvas.axes1.scatter(slider_value - 1, self.worker.example_context_tensor.cpu().numpy().flatten()[slider_value - 1], color='blue', s=100)
    
        # Update the attention weights plot based on the slider value
        self.update_slider_plot(slider_value)
        self.plot_canvas.draw()




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


