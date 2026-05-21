"""Utility helpers for CRAFT federated learning experiments."""

import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import pytz
import torch
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from omegaconf import OmegaConf

CRAFT_ALG = "craft"


def set_seed(seed):
    """Set random seeds for reproducibility across all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_cfg(arg):
    """Load and merge configuration from multiple sources with hierarchical priority."""
    arg_cmd = vars(arg)

    def _normalize_dataset_name(raw_dataset):
        """Normalize dataset names used in this codebase."""
        if raw_dataset is None:
            return None
        return str(raw_dataset).lower()

    def _parse_alg_name(raw_alg):
        """Normalize algorithm names for config lookup."""
        if not raw_alg:
            return ""
        alg = str(raw_alg).lower()
        return alg

    utils_dir = os.path.dirname(os.path.abspath(__file__))
    base_cfg_path = os.path.join(utils_dir, "config.yaml")
    cfg = OmegaConf.load(base_cfg_path)

    user_cfg = OmegaConf.load(arg.cfg)

    cfg = OmegaConf.merge(cfg, user_cfg)

    arg_cmd = {k: v for k, v in arg_cmd.items() if v is not None}
    cfg = OmegaConf.merge(cfg, OmegaConf.create(arg_cmd))

    if getattr(cfg, "dataset", None) is not None:
        cfg.dataset = _normalize_dataset_name(cfg.dataset)

    supported_datasets = {"cifar10", "cifar100", "femnist"}
    if str(getattr(cfg, "dataset", "")).lower() not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset: {cfg.dataset}. Supported datasets: cifar10, cifar100, femnist."
        )

    model_lower = str(getattr(cfg, "model", "")).lower()
    if model_lower not in {"cnn", "mlp"} and not model_lower.startswith("resnet"):
        raise ValueError(f"Unsupported model: {cfg.model}. Supported models: cnn, mlp, resnet*.")

    cfg_alg = str(getattr(cfg, "alg", "")).lower()

    cfg_alg_base = _parse_alg_name(cfg_alg)

    if cfg_alg_base != CRAFT_ALG:
        raise ValueError(f"Unsupported algorithm: {cfg.alg}. Supported algorithm: craft.")

    cfg.alg = cfg_alg_base

    return cfg


def save_config(cfg, file_name="config"):
    """Save configuration to a YAML file in the results directory."""
    path = os.path.join(cfg.dir_res, file_name + ".yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_to_save = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "rho" in cfg_to_save:
        del cfg_to_save["rho"]
    OmegaConf.save(config=cfg_to_save, f=path)


def apply_model_state(model, state):
    """Copy a state mapping into a model without checkpoint/resume APIs."""
    current_state = model.state_dict()
    with torch.no_grad():
        for key, value in state.items():
            if key not in current_state:
                raise KeyError(f"Unexpected model state key: {key}")
            target = current_state[key]
            source = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def get_logger(dir_res, log_to_console=True):
    """Create and configure a dual-output logger for experiment logging."""
    dir_log = os.path.join(dir_res, "0_log.txt")

    log = logging.getLogger(dir_log)
    log.setLevel(logging.INFO)

    if log.handlers:
        log.handlers.clear()

    file_handler = logging.FileHandler(dir_log)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(console_handler)

    log.propagate = False

    return log


def get_time_stamp(time_zone="Europe/Berlin"):
    """Generate a timezone-aware timestamp string for unique experiment identification."""
    tz = pytz.timezone(time_zone)
    test_time = datetime.now(tz).strftime("%Y%m%d-%H%M%S-%f")
    return test_time


def create_folder(cfg, dir_parent="results", prefix="test"):
    """Create a results directory for the experiment with a descriptive name."""
    if hasattr(cfg, "dir_results"):
        dir_parent = os.path.join(dir_parent, cfg.dir_results)

    results_root = getattr(cfg, "results_root", None)
    if results_root:
        if os.path.isabs(dir_parent):
            base_dir = dir_parent
        else:
            base_dir = os.path.join(results_root, dir_parent)
    else:
        base_dir = os.path.join(sys.path[0], dir_parent)

    cfg.tag = get_label(cfg)
    test_time = get_time_stamp(time_zone="Europe/Berlin")
    path_folder = f"{prefix}_{cfg.tag}_{test_time}"

    dir_results = os.path.join(base_dir, path_folder)
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    return dir_results


def select_device(cfg):
    """Auto-detect and select the best available compute device."""
    if torch.cuda.is_available():
        if "cuda" not in cfg.device:
            cfg.device = "cuda"
    elif torch.backends.mps.is_available():
        cfg.device = "mps"
    else:
        cfg.device = "cpu"


def get_label(cfg):
    """Generate a descriptive label string encoding key experiment parameters."""
    lr = "{:.0e}".format(cfg.lr).replace("e-0", "e-")

    partition = getattr(cfg, "partition", "iid")
    if partition == "dir":
        split_tag = f"dir{getattr(cfg, 'dir_alpha', 'na')}"
    else:
        split_tag = "iid"

    label = (
        f"{cfg.dataset}_{cfg.model}_{cfg.alg}_m-{cfg.m}_T-{cfg.T}_"
        f"{split_tag}_fr-{cfg.frac}_"
        f"E-{cfg.E}_bs-{cfg.bs}_lr-{lr}"
    )
    return label


def plot_acc_loss(cfg, data_path):
    """Generate and save training loss and test accuracy plots."""
    label_font = FontProperties(family="sans-serif", weight="normal", size=12)
    tick_font = FontProperties(family="sans-serif", weight="normal", size=11)
    data_dir = os.path.split(data_path)[0]

    data = np.load(data_path, allow_pickle=True).tolist()

    data["loss"] = [1e2 if x == float("inf") else x for x in data["loss"]]
    data["loss"] = [1e2 if isinstance(x, float) and (x != x) else x for x in data["loss"]]

    alg = str(getattr(cfg, "alg", "exp")).lower()
    line_color = "magenta" if alg == CRAFT_ALG else "tab:blue"
    marker = "o" if alg == CRAFT_ALG else "s"
    marker_size = 10
    linewidth = 1.5

    def _markevery_by_rounds(x_vals, round_step=50, include_rounds=(1,)):
        """Return marker indices aligned with communication rounds."""
        x_arr = np.asarray(x_vals)
        if x_arr.size == 0:
            return None
        try:
            x_float = x_arr.astype(float)
        except (TypeError, ValueError):
            return None

        indices = []
        include = tuple(float(v) for v in (include_rounds or ()))
        for idx, x_val in enumerate(x_float):
            mark = any(np.isclose(x_val, inc) for inc in include)
            if not mark and round_step:
                mark = x_val > 0 and np.isclose(np.mod(x_val, float(round_step)), 0.0)
            if mark:
                indices.append(idx)
        return indices or None

    def _plot_curve(y_values, ylabel, title, save_name, yticks=None, grid=False):
        y_plot = np.asarray(y_values[start:stop], dtype=float)
        x_plot = np.asarray(xaxis[start:stop], dtype=float)
        if y_plot.size == 0 or x_plot.size == 0:
            return

        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=80)
        ax.plot(
            x_plot,
            y_plot,
            color=line_color,
            marker=marker,
            markerfacecolor="none",
            markeredgecolor=line_color,
            markersize=marker_size,
            markevery=_markevery_by_rounds(x_plot),
            linestyle="-",
            linewidth=linewidth,
            clip_on=False,
        )
        ax.set_xlabel("Communication rounds", fontproperties=label_font, fontsize=12)
        ax.set_ylabel(ylabel, fontproperties=label_font, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        if yticks is not None:
            ax.set_yticks(yticks)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(tick_font)
        if grid:
            ax.grid(linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xlim(x_plot[0], x_plot[-1] + 1.0)
        fig.set_facecolor("white")
        plt.savefig(f"{data_dir}/{save_name}", bbox_inches="tight", dpi=300)
        plt.close(fig)

    start = 1
    rounds = None
    for key in ("round", "rounds", "round_list", "rounds_list"):
        if key in data and data[key] is not None:
            rounds = np.asarray(data[key])
            break
    if rounds is None or rounds.size == 0:
        xaxis = np.arange(len(data["loss"]))
    else:
        xaxis = rounds
    stop = min(len(xaxis), len(data["loss"]), len(data["accu_test"]))

    _plot_curve(data["loss"], "Loss", "Training loss", "res_loss.jpg")
    _plot_curve(
        data["accu_test"],
        "Test accuracy",
        "Test accuracy",
        "res_accu.jpg",
        grid=True,
    )
