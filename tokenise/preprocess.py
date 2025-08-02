# heartgpt/tokenise/preprocess.py

import numpy as np


def tokenise_biosignal(data, max_length=500):
    # ensure data shape is (channels, time)
    if data.ndim > 1 and data.shape[0] > data.shape[1]:
        data = data.T

    if data.shape[1] > max_length:
        data = data[:, -max_length :]

    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:
        scaled = np.zeros_like(data)
    else:
        scaled = (data - data_min) / (data_max - data_min)

    scaled *= 100
    rounded = np.round(scaled)
    return rounded.astype(int)
