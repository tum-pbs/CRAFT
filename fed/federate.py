"""Federated learning training orchestration module."""

import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fed.dataset import DatasetSplit
from utils.utils import get_logger, plot_acc_loss


def FL_train(cfg, server, clients, dataset):
    """Run the complete federated learning training loop."""
    res_dict, res_path, log = prepare(cfg, clients, server, dataset)
    train_start_time = time.time()
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(train_start_time))
    log.info(f"\n-----Start training ({cfg.device})-----")
    # log.info(f"{local_time}")
    for t in range(0, cfg.T + 1):

        # Select the clients participating in this round.
        selected_clients = np.asarray(server.select_clients(frac=cfg.frac), dtype=np.int64)

        # Let selected clients train locally and collect their updates.
        res_clients = clients.local_update(server.model, selected_clients, t)

        # Aggregate client updates into the global model.
        server.aggregate(res_clients)

        # Periodically evaluate the global model and save metrics.
        if t % cfg.save_freq == 0 or t in [1, cfg.T]:
            run_time = (time.time() - train_start_time) / 60
            loss = evaluate(cfg, server, dataset, res_dict, log, round_t=t, run_time=run_time)
            if cfg.plot:
                plot_acc_loss(cfg, res_path)
            if np.isnan(loss):
                log.info(f"Early stop due to the loss explosion at round {t}.")
                break
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    log.info(f"-----End training-----")
    # log.info(f"{local_time}")


def prepare(cfg, clients, server, dataset):
    """Prepare for training: set up logging and results dictionary."""
    res_path = os.path.join(cfg.dir_res, f"0_results.npy")
    log = get_logger(cfg.dir_res)
    res_dict = defaultdict(list)
    evaluate(cfg, server, dataset, res_dict, log, round_t=0, run_time=0.0)
    clients.res_dict = res_dict
    clients.log = log
    return res_dict, res_path, log


def evaluate(cfg, server, dataset, res_dict, log, round_t, run_time=None):
    """Evaluate the global model, then log and save metrics."""
    loss = cal_loss(cfg, server.model, dataset)
    client_accus = cal_client_test_accu(cfg, server.model, dataset)
    acc_test = float(np.mean(client_accus))
    acc_std = float(np.std(client_accus))
    res_dict["round"].append(round_t)
    res_dict["loss"].append(loss)
    res_dict["accu_clients_test"].append(client_accus)
    res_dict["accu_test"].append(acc_test)
    res_dict["accu_test_std"].append(acc_std)
    if round_t != 0:
        time_text = ""
        if run_time is not None:
            time_text = f" | Total time: {run_time:>6.1f} min"
        log.info(
            f"Round: {round_t:>4d} | "
            f"Loss: {loss:>7.3f} | "
            f"Accu: {acc_test:>7.3f} | "
            f"Std: {acc_std:>7.3f}"
            f"{time_text}"
        )
    save_results(cfg, res_dict)
    return loss


def save_results(cfg, res_dict):
    """Save evaluation metrics to disk."""
    res_path = os.path.join(cfg.dir_res, f"0_results.npy")
    np.save(res_path, np.asarray(res_dict, dtype=object))


def move_batch_to_device(batch, device):
    """Move a DataLoader batch to the specified device."""
    inputs, labels = batch
    data = inputs.to(device)
    labels = labels.to(device)
    return data, labels


def cal_loss(cfg, model, dataset, bs=1000):
    """Compute training loss over the dataset."""
    total_loss = 0.0
    total_samples = 0
    model.eval()
    train_split = dataset["split_dict"]["train"]
    train_indices = np.concatenate([np.asarray(v, dtype=np.int64) for v in train_split.values()])
    train_dataset = Subset(dataset["train"], train_indices.tolist())
    data_loader = DataLoader(train_dataset, batch_size=bs)
    with torch.no_grad():
        for batch in data_loader:
            data, label = move_batch_to_device(batch, cfg.device)
            preds = model(data)
            batch_loss = model.loss(preds, label)
            batch_size = label.shape[0] if hasattr(label, "shape") else len(label)
            total_loss += batch_loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def cal_accu(cfg, model, dataset, bs=1000):
    """Compute model accuracy over a dataset."""
    correct = 0
    model.eval()
    data_loader = DataLoader(dataset, batch_size=bs)
    with torch.no_grad():
        for batch in data_loader:
            data, label = move_batch_to_device(batch, cfg.device)
            pred = model(data)
            pred = pred.argmax(dim=1)
            correct += (pred.view_as(label) == label).sum()
        accuracy = correct / len(dataset)
    return accuracy.item()


def cal_client_test_accu(cfg, model, dataset):
    """Compute per-client test accuracies."""
    client_accus = np.empty(cfg.m, dtype=float)
    for client_i in range(cfg.m):
        client_test = DatasetSplit(dataset, client_i, split="test")
        client_accus[client_i] = cal_accu(cfg, model, client_test)
    return client_accus
