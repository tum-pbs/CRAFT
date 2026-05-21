"""Federated learning client module."""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from fed.dataset import DatasetSplit
from fed.federate import move_batch_to_device


class Clients:
    """Encapsulates all client-side responsibilities for federated learning."""

    def __init__(self, cfg, dataset):
        """Initialize the Clients object with configuration and shared resources."""
        self.cfg = cfg
        self.device = cfg.device
        self.datasets = dataset

    def local_update(self, model_server, selected_clients, round_t):
        """Run local training for each selected client and collect their updates."""
        selected_clients = [int(i) for i in selected_clients]
        res_clients = {"models": {}}

        # Local training for each selected client.
        for client_i in selected_clients:
            model_i = copy.deepcopy(model_server).train()
            dataset_i = DatasetSplit(self.datasets, client_i)
            data_loader = init_data_loader(self.cfg, dataset_i)
            optimizer = init_optimizer(self.cfg, model_i, round_t)
            for _ in range(int(self.cfg.E)):  # Number of local epochs.
                for batch in data_loader:
                    data, labels = move_batch_to_device(batch, self.device)
                    model_i.zero_grad()
                    pred = model_i(data)
                    loss = model_i.loss(pred, labels)
                    loss.backward()
                    optimizer.step()
            res_clients["models"][client_i] = model_i.state_dict()
        res_clients["selected_clients"] = selected_clients
        return res_clients


def init_optimizer(cfg, model, t):
    """Initialize the optimizer for local client training."""
    lr = resolve_round_lr(cfg, t)
    return torch.optim.SGD(model.parameters(), lr=lr)


def init_data_loader(cfg, dataset_i, shuffle=True):
    """Initialize a DataLoader for local client training."""
    dataset_len = len(dataset_i)
    if cfg.bs:
        data_loader = DataLoader(dataset_i, batch_size=cfg.bs, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset_i, batch_size=dataset_len, shuffle=shuffle)
    return data_loader


def resolve_round_lr(cfg, t):
    """Compute the learning rate for a specific communication round."""
    if cfg.lr_decay:
        decay_base = float(cfg.lr_decay)
        decay_step = max(int(t) - 1, 0)
        return cfg.lr * np.power(decay_base, decay_step)
    return cfg.lr
