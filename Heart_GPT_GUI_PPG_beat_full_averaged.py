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
from scipy.signal import find_peaks
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


# Assuming the model and other configurations are already set up
model_config = 'PPG_PT'
block_size = 500
shift_parameter = 100
batch_size = 150
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
# change model directory here to where you have the models stored
model_path_ppg = "D:/HeartGPTModels/PPGPT_beat_5k_iters.pth"
# model_path_ppg_AF = "D:/HeartGPTModels/PPGPT_AF_1k_iters.pth"

model_path_ecg = "D:/HeartGPTModels/ECGPT_560k_iters.pth"
# model_path_ecg_AF = "D:/HeartGPTModels/ECGPT_AF_1k_iters.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_config == 'PPG_PT':
    vocab_size = 102
    model_path = model_path_ppg
    # model_path_AF = model_path_ppg_AF
elif model_config == 'ECG_PT':
    vocab_size = 101
    model_path = model_path_ecg
    # model_path_AF = model_path_ecg_AF

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
        return logits


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes1 = fig.add_subplot(211)  # First plot (Tokenised Input Data and Generated Output)
        self.axes2 = fig.add_subplot(212)  # Second plot (Model Output)
        super(PlotCanvas, self).__init__(fig)
        fig.tight_layout(pad=3.0)  # Add padding between plots

        self.toolbar = NavigationToolbar(self, parent)
        parent.layout.addWidget(self.toolbar)  # Add padding between plots
        
        # Link the x-axes
        self.axes1.callbacks.connect('xlim_changed', self.on_xlim_changed)
        self.axes2.callbacks.connect('xlim_changed', self.on_xlim_changed)

    def on_xlim_changed(self, ax):
        if ax == self.axes1:
            self.axes2.set_xlim(self.axes1.get_xlim())
        elif ax == self.axes2:
            self.axes1.set_xlim(self.axes2.get_xlim())
        self.draw()

    def plot(self, input_data=None, output_data=None, context_length=None, num_samples=None):
        if input_data is not None and output_data is not None and context_length is not None and num_samples is not None:
            self.axes1.clear()
            self.axes1.plot(input_data.flatten()[:num_samples], color='black', label='Tokenised Input Data')
            if len(output_data) > context_length:
                self.axes1.plot(range(context_length, context_length + len(output_data) - context_length), output_data[context_length:num_samples], color='red')
            self.axes1.set_xlabel('Token Number')
            self.axes1.set_ylabel('Token\nValue')
            self.axes1.legend()

            output_data = np.array(output_data)[:num_samples]

            peaks, _ = find_peaks(output_data, height=0.015, width=(None, 20), distance=14)
            peaks = peaks.astype(int)
            for peak in peaks:
                peak_value = output_data[peak]
                if peak_value < 0.2051:
                    color = 'red'
                elif 0.2051 <= peak_value <= 0.4598:
                    color = 'orange'
                else:
                    color = 'green'
                self.axes1.plot(peak, input_data.flatten()[peak], 'x', color=color, markersize=10)

            self.axes2.clear()
            self.axes2.plot(output_data, color='blue', label='Model Output')
            self.axes2.set_xlabel('Index')
            self.axes2.set_ylabel('Value')
            self.axes2.legend()

        self.draw()










class Worker(QThread):
    finished_signal = pyqtSignal(list)

    def __init__(self, model, example_context_tensor):
        super().__init__()
        self.model = model
        self.example_context_tensor = example_context_tensor
        self.input_stitched = stitch_with_overlap(example_context_tensor.cpu().numpy())
        self.model_output = None  # Add this line

    def run(self):
        outputs_list = []
        X_input = self.example_context_tensor
        for i in range(0, len(X_input), batch_size):
            # Get the current batch
            batch = X_input[i:i+batch_size]

            # Run the batch through the model
            with torch.no_grad():
                outputs = self.model(batch)

            # Append the outputs, labels, and inputs to their respective lists
            outputs_list.append(outputs.detach().cpu().numpy())

            #outputs_list.append(outputs)

            #print(f"Batch {i//150 + 1}")


        outputs_np = np.concatenate(outputs_list, axis=0)
        #print(outputs_np.shape[0])
        #print(outputs_np.shape[1])
        outputs_np_reshape = outputs_np.reshape(outputs_np.shape[0],outputs_np.shape[1])
        outputs_stitched = stitch_with_overlap(outputs_np_reshape)
        self.model_output = outputs_stitched.tolist()
        self.finished_signal.emit(self.model_output)





class App(QMainWindow):
    update_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.title = 'HeartGPT: PPG Beat Detection (Finger PPG Only)'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.model_config = 'PPG_PT'  # Always use PPG_PT
        self.initUI()
        self.first_model_done = False

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.plot_canvas = PlotCanvas(self)
        self.layout.addWidget(self.plot_canvas)

        form_layout = QFormLayout()

        self.layout.addLayout(form_layout)

        self.button = QPushButton('Load Context and Estimate Beats', self)  # Updated button text
        self.button.clicked.connect(self.load_and_plot_data)
        self.layout.addWidget(self.button)

        self.save_button = QPushButton('Save Beats', self)  # Updated button text
        self.save_button.clicked.connect(self.save_attention_weights)
        self.layout.addWidget(self.save_button)

        self.progress_label = QLabel('Ready', self)
        self.layout.addWidget(self.progress_label)

        self.probability_label = QLabel('Average Estimated Beat Confidence: ', self)
        self.layout.addWidget(self.probability_label)


    # Remove update_model_config method
    # def update_model_config(self):
    #     selected_model = self.model_selector.currentText()
    #     if selected_model == "PPGPT":
    #         self.model_config = 'PPG_PT'
    #     elif selected_model == "ECGPT":
    #         self.model_config = 'ECG_PT'
    #     self.update_model_parameters()

    def update_model_parameters(self):
        global vocab_size, model_path
        if self.model_config == 'PPG_PT':
            vocab_size = 102
            model_path = model_path_ppg
        # Remove ECG_PT configuration
        # elif self.model_config == 'ECG_PT':
        #     vocab_size = 101
        #     model_path = model_path_ecg

    def save_attention_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder:
            output_file = f"{folder}/output_beats.mat"

            # Assuming context and model output are stored in self.context and self.model_output
            context = self.worker.example_context_tensor.cpu().numpy()
            averaged_context = stitch_with_overlap(context)[:self.num_samples]  # Use self.num_samples
            model_output = np.array(self.worker.model_output)[:self.num_samples]  # Use self.num_samples

            # Find peaks in the model output
            peaks, properties = find_peaks(model_output, height=0.015, width=(None, 20), distance=14)
            peaks_array = np.zeros_like(model_output)
            peaks_array[peaks] = properties['peak_heights']

            # Save the context, model output, and peaks array in a .mat file
            savemat(output_file, {'context': averaged_context, 'model_output': model_output, 'peaks_array': peaks_array})

            self.progress_label.setText(f'Saved to {output_file}')




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
            global weights_matrices
            weights_matrices = []

            df = pd.read_csv(input_file, header=None)
            input_data = df.values

            # Convert input_data to Nx500 array with 100-sample shifts
            self.num_samples = input_data.shape[0]  # Store num_samples as an instance variable
            reshaped_data = []
            for i in range(0, self.num_samples, shift_parameter):
                segment = input_data[i:i+500]
                segment = tokenize_biosignal(segment).flatten()
                if len(segment) < 500:
                    segment = np.pad(segment, (0, 500 - len(segment)), 'constant')
                reshaped_data.append(segment)
            reshaped_data = np.array(reshaped_data)
            reshaped_data = np.nan_to_num(reshaped_data)

            self.model_config = 'PPG_PT'  # Always use PPG_PT
            plot_title = "PPGPT Output"
            self.update_model_parameters()

            model = HeartGPTModel()
            model.lm_head = NewHead(n_embd)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            model.to(device)

            example_context_tensor = torch.tensor(reshaped_data, dtype=torch.long, device=device)

            averaged_input = stitch_with_overlap(reshaped_data)
            averaged_input = averaged_input[:self.num_samples]

            self.plot_canvas.axes1.set_title(plot_title)
            self.plot_canvas.plot(input_data=averaged_input.flatten(), output_data=[], context_length=averaged_input.shape[0], num_samples=self.num_samples)

            self.worker = Worker(model, example_context_tensor)
            self.worker.finished_signal.connect(self.plot_output)
            self.worker.start()

            self.save_button.setEnabled(False)





    def plot_output(self, output):
        self.worker.model_output = output
        averaged_input = stitch_with_overlap(self.worker.example_context_tensor.cpu().numpy())
        num_samples = averaged_input.shape[0]
        self.plot_canvas.plot(input_data=averaged_input.flatten()[:self.num_samples], output_data=output, context_length=averaged_input.shape[0], num_samples=self.num_samples)
        self.plot_canvas.axes2.clear()
        self.plot_canvas.axes2.plot(output[:self.num_samples], color='blue', label='Model Output')
        self.plot_canvas.axes2.set_xlabel('Index')
        self.plot_canvas.axes2.set_ylabel('Value')
        self.plot_canvas.axes2.legend()
        self.plot_canvas.draw()
        self.progress_label.setText('Processing complete')
        self.save_button.setEnabled(True)

        output_data = np.array(output)[:self.num_samples]
        peaks, _ = find_peaks(output_data, height=0.015, width=(None, 20), distance=14)
        peak_values = output_data[peaks]
        median_peak_value = np.median(peak_values)

        if median_peak_value < 0.2051:
            category = "Low"
            color = "red"
        elif 0.2051 <= median_peak_value <= 0.4598:
            category = "Medium"
            color = "orange"
        else:
            category = "High"
            color = "green"

        self.probability_label.setText(f'Average Estimated Beat Confidence: <span style="color:{color}">{category}</span>')









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

def stitch_with_overlap(arr):
    N, width = arr.shape
    overlap = 500-shift_parameter  # Since each window overlaps by 400 samples (500 - shift)
    step = width - overlap
    total_length = step * (N - 1) + width
    result = np.zeros(total_length)
    counts = np.zeros(total_length)

    for i in range(N):
        start = i * step
        end = start + width
        result[start:end] += arr[i]
        counts[start:end] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    result /= counts

    return result

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
