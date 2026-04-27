"""
@author: Gaetan Hadjeres
"""

import torch


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()


def init_hidden(num_layers, batch_size, lstm_hidden_size):
    hidden = (
        cuda_variable(torch.randn(num_layers, batch_size, lstm_hidden_size)),
        cuda_variable(torch.randn(num_layers, batch_size, lstm_hidden_size))
    )
    return hidden
