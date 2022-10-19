import os
import random
from collections import OrderedDict
from typing import List, Optional, Sequence
import einops
import numpy as np
import torch
import torch.nn as nn
import pynvml

from torch.utils.data import random_split
import wandb
from prettytable import PrettyTable
import logging


pynvml.nvmlInit()

# Modified from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return total_params, table


def get_split_idx(l, seed, train_fraction=0.95):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class eval_mode:
    def __init__(self, *models, no_grad=False):
        self.models = models
        self.no_grad = no_grad
        self.no_grad_context = torch.no_grad()

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)
        if self.no_grad:
            self.no_grad_context.__enter__()

    def __exit__(self, *args):
        if self.no_grad:
            self.no_grad_context.__exit__(*args)
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def batch_indexing(input, idx):
    """
    Given an input with shape (*batch_shape, k, *value_shape),
    and an index with shape (*batch_shape) with values in [0, k),
    index the input on the k dimension.
    Returns: (*batch_shape, *value_shape)
    """
    batch_shape = idx.shape
    dim = len(idx.shape)
    value_shape = input.shape[dim + 1 :]
    N = batch_shape.numel()
    assert input.shape[:dim] == batch_shape, "Input batch shape must match index shape"
    assert len(value_shape) > 0, "No values left after indexing"

    # flatten the batch shape
    input_flat = input.reshape(N, *input.shape[dim:])
    idx_flat = idx.reshape(N)
    result = input_flat[np.arange(N), idx_flat]
    return result.reshape(*batch_shape, *value_shape)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)


def cuda_free_mem(gpu_index: int) -> float:
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free


def cuda_idxmax_free_mem() -> int:
    gpu_count = pynvml.nvmlDeviceGetCount()
    free_mems = [cuda_free_mem(i) for i in range(gpu_count)]
    return int(np.argmax(free_mems))


class TrainWithLogger:
    def reset_log(self):
        self.log_components = OrderedDict()

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value.detach().cpu().item()),
            )

    def flush_log(self, epoch, iterator=None):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        if iterator is not None:
            iterator.set_postfix_str(postfix)
        wandb.log(log_components, step=epoch)
        self.log_components = OrderedDict()


class SaveModule(nn.Module):
    def set_snapshot_path(self, path):
        self.snapshot_path = path
        print(f"Setting snapshot path to {self.snapshot_path}")

    def save_snapshot(self):
        os.makedirs(self.snapshot_path, exist_ok=True)
        torch.save(self.state_dict(), self.snapshot_path / "snapshot.pth")

    def load_snapshot(self):
        self.load_state_dict(torch.load(self.snapshot_path / "snapshot.pth"))


def split_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set
