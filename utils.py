import numpy as np
import torch


def de_norm(tensor):
    mean = tensor.new_tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def make_numpy_grid(tensor, nrow=4):
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    count, channels, height, width = tensor.shape
    nrow = min(nrow, count)
    ncol = (count + nrow - 1) // nrow
    grid = tensor.new_zeros(channels, ncol * height, nrow * width)
    for index, item in enumerate(tensor):
        row, col = divmod(index, nrow)
        grid[:, row * height:(row + 1) * height, col * width:(col + 1) * width] = item
    return grid.permute(1, 2, 0).numpy()
