"""Federated learning dataset module."""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


def _repo_root():
    """Return the repository root regardless of the current working directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dataset_root(cfg=None):
    """Resolve the default dataset cache directory under the repository root."""
    configured_root = None
    if cfg is not None:
        configured_root = getattr(cfg, "dataset_root", None)
        if configured_root is None:
            configured_root = getattr(cfg, "data_root", None)

    if configured_root:
        configured_root = os.path.expandvars(os.path.expanduser(str(configured_root)))
        if not os.path.isabs(configured_root):
            configured_root = os.path.join(_repo_root(), configured_root)
        return os.path.abspath(configured_root)

    return os.path.join(_repo_root(), "dataset")


def _resolve_partition(cfg):
    """Resolve and normalize the partition type from configuration."""
    partition = str(getattr(cfg, "partition", "") or "").lower()
    if not partition:
        partition = "iid"
    if partition in {"dirichlet", "dir"}:
        return "dir"
    if partition in {"iid"}:
        return "iid"
    raise ValueError(f"Unknown partition type: {partition}")


def _normalize_dataset_name(dataset_name):
    """Normalize dataset aliases to canonical internal names."""
    if dataset_name is None:
        return dataset_name
    return str(dataset_name).lower()


def _as_numpy(array_like):
    """Convert a tensor or array-like object to a NumPy array."""
    if torch.is_tensor(array_like):
        return array_like.cpu().numpy()
    return np.array(array_like)


def _extract_arrays(dataset):
    """Extract raw data and label arrays from various dataset types."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = np.asarray(dataset.indices)
        data, labels = _extract_arrays(base)
        return data[indices], labels[indices]

    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        data = _as_numpy(dataset.data)
        labels = _as_numpy(dataset.targets)
        return data, labels.astype(np.int64)

    if hasattr(dataset, "data") and hasattr(dataset, "labels"):
        data = _as_numpy(dataset.data)
        labels = _as_numpy(dataset.labels)
        return data, labels.astype(np.int64)

    raise ValueError("Unsupported dataset type for array extraction.")


def _split_iid_indices(labels, num_clients, rng):
    """Partition data indices with an IID strategy."""
    indices = np.arange(len(labels))
    # Shuffle once globally, then split evenly so each client receives a random IID slice.
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {i: np.array(splits[i], dtype=np.int64) for i in range(num_clients)}


def _split_dirichlet_indices(
    labels,
    num_clients,
    num_classes,
    alpha,
    rng,
    balance=True,
    max_retry=200,
):
    """Partition data indices with a Dirichlet distribution for non-IID heterogeneity."""
    min_size = 0
    num_samples = len(labels)
    # In balanced mode, aim for at least one sample per class when feasible.
    raw_min_target = num_classes if balance else 1
    max_feasible_target = num_samples // max(1, int(num_clients))
    min_target = min(raw_min_target, max_feasible_target)
    if min_target < raw_min_target:
        print(
            "[Dirichlet][Warn] Requested minimum per-client samples is infeasible; "
            f"clamped from {raw_min_target} to {min_target}."
        )

    max_retry = max(1, int(max_retry))
    attempts = 0
    best_idx_batch = None
    best_min_size = -1

    while min_size < min_target and attempts < max_retry:
        attempts += 1
        idx_batch = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)

            # Dirichlet alpha controls label skew; smaller values concentrate a class.
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))

            if balance:
                # Downweight clients that already exceed the average target size.
                proportions = np.array(
                    [
                        p * (len(idx_j) < num_samples / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                prop_sum = proportions.sum()
                if prop_sum > 0:
                    proportions = proportions / prop_sum
                else:
                    proportions = np.repeat(1.0 / num_clients, num_clients)

            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_split = np.split(idx_k, split_points)

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_split)]

        min_size = min(len(idx_j) for idx_j in idx_batch)
        if min_size > best_min_size:
            best_min_size = min_size
            best_idx_batch = [idx_j.copy() for idx_j in idx_batch]

    if min_size < min_target:
        if best_idx_batch is not None:
            idx_batch = best_idx_batch
            min_size = best_min_size

        if min_target > 0:
            idx_dict = {i: np.array(idx_batch[i], dtype=np.int64) for i in range(num_clients)}
            idx_dict = _enforce_min_total_samples_per_client(
                idx_dict,
                min_total=min_target,
                rng=rng,
            )
            idx_batch = [idx_dict[i].tolist() for i in range(num_clients)]

    if balance:
        target = num_samples // num_clients
        # Trim oversized clients first; trimmed samples are redistributed to small clients.
        extras = []
        for i in range(num_clients):
            idxs = idx_batch[i]
            if len(idxs) > target:
                rng.shuffle(idxs)
                extras.extend(idxs[target:])
                idx_batch[i] = idxs[:target]

        rng.shuffle(extras)

        for i in range(num_clients):
            need = target - len(idx_batch[i])
            if need > 0:
                take = extras[:need]
                idx_batch[i] = idx_batch[i] + take
                extras = extras[need:]

        for i in range(num_clients):
            if not extras:
                break
            idx_batch[i] = idx_batch[i] + [extras.pop()]

    return {i: np.array(idx_batch[i], dtype=np.int64) for i in range(num_clients)}


def _enforce_min_total_samples_per_client(client_indices, min_total, rng):
    """Enforce a hard lower bound on total samples per client."""
    min_total = int(min_total)
    if min_total <= 0:
        return {k: np.asarray(v, dtype=np.int64) for k, v in client_indices.items()}

    client_ids = sorted(client_indices.keys())
    m = len(client_ids)

    buckets = {}
    total_samples = 0
    for cid in client_ids:
        arr = np.asarray(client_indices[cid], dtype=np.int64).copy()
        rng.shuffle(arr)
        buckets[cid] = arr.tolist()
        total_samples += len(arr)

    required = m * min_total
    if total_samples < required:
        raise ValueError(
            "Hard constraint infeasible: total samples "
            f"{total_samples} < num_clients * min_total ({m} * {min_total} = {required})."
        )

    deficits = [cid for cid in client_ids if len(buckets[cid]) < min_total]
    for cid in deficits:
        need = min_total - len(buckets[cid])
        while need > 0:
            # Always draw from the largest legal donor to preserve the lower bound globally.
            donor = None
            donor_size = -1
            for d in client_ids:
                size_d = len(buckets[d])
                if size_d > min_total and size_d > donor_size:
                    donor = d
                    donor_size = size_d

            if donor is None:
                raise RuntimeError(
                    "Failed to satisfy hard min_total constraint despite sufficient "
                    "global sample count."
                )

            movable = len(buckets[donor]) - min_total
            take = min(need, movable)
            moved = buckets[donor][-take:]
            buckets[donor] = buckets[donor][:-take]
            buckets[cid].extend(moved)
            need -= take

    return {cid: np.asarray(buckets[cid], dtype=np.int64) for cid in client_ids}


def split_client_train_test(client_indices, train_ratio, rng):
    """Split each client's data indices into local train and test sets."""
    train_dict = {}
    test_dict = {}
    for client, idxs in client_indices.items():
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        split = int(train_ratio * len(idxs))
        if len(idxs) > 0 and train_ratio > 0.0 and split == 0:
            # Avoid clients with data contributing nothing to local training.
            split = 1
        train_dict[client] = idxs[:split]
        test_dict[client] = idxs[split:]
    return train_dict, test_dict


class ArrayDataset(Dataset):
    """PyTorch Dataset wrapper for array-backed vision data with optional transforms."""

    def __init__(self, data, labels, transform=None):
        """Initialize ArrayDataset with image data, labels, and optional transform."""
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Retrieve a single sample by index."""
        img = self.data[idx]
        label = int(self.labels[idx])
        if torch.is_tensor(img):
            img = img.numpy()
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img.astype("uint8"), "L")
            else:
                img = Image.fromarray(img.astype("uint8"))
        if self.transform:
            img = self.transform(img)
        return img, label


def init_dataset(cfg, min_total_samples_per_client=20, client_train_ratio=0.8):
    """Initialize and partition a federated learning dataset based on configuration."""
    cfg.dataset = _normalize_dataset_name(getattr(cfg, "dataset", None))
    path = _dataset_root(cfg)

    if cfg.dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        trans_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        trans_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset_train = datasets.CIFAR10(root=path, train=True, download=True)
        dataset_test = datasets.CIFAR10(root=path, train=False, download=True)
        cfg.num_classes = 10

    elif cfg.dataset == "cifar100":
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        trans_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        trans_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset_train = datasets.CIFAR100(root=path, train=True, download=True)
        dataset_test = datasets.CIFAR100(root=path, train=False, download=True)
        cfg.num_classes = 100

    elif cfg.dataset == "femnist":
        dataset_train, dataset_test, split_dict = load_FEMNIST_data_hf(cfg)
        cfg.num_classes = 62
        rng = np.random.default_rng(getattr(cfg, "seed", None))
        # FEMNIST split_dict is already writer/client based; only split within each writer.
        train_split, test_split = split_client_train_test(split_dict, client_train_ratio, rng)
        dataset = {
            "train": dataset_train,
            "test": dataset_train,
            "split_dict": {"train": train_split, "test": test_split},
        }
        cfg.data_train = sum(len(v) for v in train_split.values())
        cfg.data_test = sum(len(v) for v in test_split.values())
        cfg.rho = cal_rho(cfg, dataset)
        return dataset

    else:
        raise ValueError(
            f"Invalid dataset: {cfg.dataset}. Supported datasets: cifar10, cifar100, femnist."
        )

    cfg.data_train = len(dataset_train)
    cfg.data_test = len(dataset_test)

    partition = _resolve_partition(cfg)
    rng = np.random.default_rng(getattr(cfg, "seed", None))
    # Merge official train/test pools before client partitioning, then create local splits.
    train_data, train_labels = _extract_arrays(dataset_train)
    test_data, test_labels = _extract_arrays(dataset_test)
    all_data = np.concatenate([train_data, test_data], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    dataset_train = ArrayDataset(all_data, all_labels, transform=trans_train)
    dataset_test = ArrayDataset(all_data, all_labels, transform=trans_test)

    if partition == "iid":
        split_dict = _split_iid_indices(all_labels, cfg.m, rng)
    elif partition == "dir":
        alpha = float(getattr(cfg, "dir_alpha", getattr(cfg, "Diralpha", 0.1)))
        balance = bool(getattr(cfg, "balance", True))
        split_dict = _split_dirichlet_indices(
            all_labels, cfg.m, cfg.num_classes, alpha, rng, balance=balance
        )
    else:
        raise ValueError(f"Unsupported partition: {partition}. Supported partitions: iid, dir.")

    split_dict = _enforce_min_total_samples_per_client(
        split_dict,
        min_total=min_total_samples_per_client,
        rng=rng,
    )

    train_split, test_split = split_client_train_test(split_dict, client_train_ratio, rng)

    dataset = {
        "train": dataset_train,
        "test": dataset_test,
        "split_dict": {
            "train": train_split,
            "test": test_split,
        },
    }

    cfg.data_train = sum(len(v) for v in train_split.values())
    cfg.data_test = sum(len(v) for v in test_split.values())
    cfg.rho = cal_rho(cfg, dataset)
    return dataset


class DatasetSplit(Dataset):
    """Lightweight Dataset view exposing only samples belonging to a specific client."""

    def __init__(self, dataset, client_i, split="train"):
        """Initialize a client-specific dataset view."""
        client_i = int(client_i)
        split_dict = dataset.get("split_dict", {})
        self.dataset = dataset.get(split, dataset.get("train"))
        if split == "test" and "test_split_dict" in dataset:
            # Backward compatibility for older saved datasets that used test_split_dict.
            indices = dataset["test_split_dict"].get(client_i, [])
        elif isinstance(split_dict, dict) and split in split_dict:
            indices = split_dict[split].get(client_i, [])
        else:
            indices = split_dict.get(client_i, [])
        self.indices_i = list(indices)

    def __len__(self):
        """Return the number of samples belonging to this client."""
        return len(self.indices_i)

    def __getitem__(self, index):
        """Retrieve a sample by local index."""
        data_index = self.indices_i[index]
        data, label = self.dataset[data_index]
        return data, label


def get_subset(dataset, set_size, classes):
    """Create a class-balanced subset of a dataset."""
    num_per_class = set_size // classes
    class_indices = {i: [] for i in range(classes)}
    for idx, data in enumerate(dataset):
        label = int(data[1])
        if len(class_indices[label]) < num_per_class:
            class_indices[label].append(idx)
        if all(len(indices) == num_per_class for indices in class_indices.values()):
            break
    subset_indices = sum(class_indices.values(), [])
    return Subset(dataset, subset_indices)


def cal_rho(cfg, dataset):
    """Calculate normalized per-client data ratios (rho) for weighted aggregation."""
    split_dict = dataset.get("split_dict", {})
    if isinstance(split_dict, dict) and "train" in split_dict:
        train_dict = split_dict["train"]
        total = sum(len(v) for v in train_dict.values())
        return [(len(train_dict.get(i, [])) / total if total > 0 else 0.0) for i in range(cfg.m)]

    total = sum(len(v) for v in split_dict.values()) if isinstance(split_dict, dict) else 0
    if total == 0:
        return [0.0 for _ in range(cfg.m)]
    return [(len(split_dict.get(i, [])) / total) for i in range(cfg.m)]


class FEMNISTDataset(Dataset):
    """PyTorch Dataset for FEMNIST (Federated Extended MNIST) handwriting data."""

    def __init__(self, data, labels, transform=None):
        """Initialize FEMNIST dataset with images, labels, and optional transform."""
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single sample by index."""
        image = self.data[idx]
        label = self.labels[idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"), "L")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_FEMNIST_data_hf(cfg):
    """Load FEMNIST from Hugging Face with natural writer-based client splits."""
    force_reload = getattr(cfg, "force_reload_femnist", False)
    cache_dir = os.path.join(_dataset_root(cfg), "femnist")
    processed_root = os.path.join(cache_dir, "processed")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(processed_root, exist_ok=True)

    sig_items = {
        "m": getattr(cfg, "m", None),
        "seed": getattr(cfg, "seed", None),
        "min": getattr(cfg, "min_samples_per_user", getattr(cfg, "min_samples", 0)),
        "max": getattr(cfg, "max_samples_per_user", getattr(cfg, "max_samples", 500)),
    }
    # Cache key must include all writer-selection inputs to avoid mixing experiments.
    signature = "_".join([f"{k}{v}" for k, v in sig_items.items()])
    processed_dir = os.path.join(processed_root, signature)

    split_path = os.path.join(processed_dir, "split_dict.json")
    meta_path = os.path.join(processed_dir, "meta.yaml")
    np_files = {
        "train_x": os.path.join(processed_dir, "train_x.npy"),
        "train_y": os.path.join(processed_dir, "train_y.npy"),
        "test_x": os.path.join(processed_dir, "test_x.npy"),
        "test_y": os.path.join(processed_dir, "test_y.npy"),
    }

    def _all_exist():
        """Return whether all cache files exist."""
        return all(os.path.isfile(p) for p in np_files.values()) and os.path.isfile(split_path)

    if (not force_reload) and _all_exist():
        try:
            print("[FEMNIST][Cache] Trying to load processed data from cache: " f"{processed_dir}")

            with open(split_path, "r") as fjs:
                split_dict_raw = json.load(fjs)
            split_dict = {int(k): np.array(v, dtype="int32") for k, v in split_dict_raw.items()}

            train_array = np.load(np_files["train_x"])
            train_labels_array = np.load(np_files["train_y"]).astype(np.int64)

            test_array = (
                np.load(np_files["test_x"])
                if os.path.isfile(np_files["test_x"])
                else np.empty((0, 28, 28), dtype=np.uint8)
            )
            test_labels_array = (
                np.load(np_files["test_y"]).astype(np.int64)
                if os.path.isfile(np_files["test_y"])
                else np.empty((0,), dtype=np.int64)
            )

            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

            dataset_train = FEMNISTDataset(train_array, train_labels_array, transform_train)
            dataset_test = FEMNISTDataset(test_array, test_labels_array, transform_eval)

            cfg.data_train = len(dataset_train)
            cfg.data_test = len(dataset_test)

            if cfg.data_test > 0:
                _, counts = np.unique(test_labels_array, return_counts=True)
                props = counts / cfg.data_test
                print(
                    "[FEMNIST][Cache] Test label distribution: "
                    f"min={np.min(props):.4f}, "
                    f"max={np.max(props):.4f}, "
                    f"std={np.std(props):.4f}"
                )

            return dataset_train, dataset_test, split_dict
        except Exception as e:
            print(f"[FEMNIST][Cache] Failed to load from cache: {e}")

    print(f"[HF] Loading flwrlabs/femnist via datasets.load_dataset ... cache_dir={cache_dir}")
    ds = load_dataset("flwrlabs/femnist", split="train", cache_dir=cache_dir)

    images = ds["image"]
    labels = ds["character"]
    writer_ids = ds["writer_id"]

    writer_to_indices = defaultdict(list)
    for idx, w in enumerate(writer_ids):
        writer_to_indices[w].append(idx)

    min_samples = getattr(cfg, "min_samples_per_user", 0)
    max_samples = getattr(cfg, "max_samples_per_user", 500)

    filtered_writers = [
        w for w, inds in writer_to_indices.items() if min_samples <= len(inds) <= max_samples
    ]
    print(
        f"[HF] total writers={len(writer_to_indices)}, "
        f"filtered={len(filtered_writers)} "
        f"(range {min_samples}-{max_samples})"
    )

    if len(filtered_writers) == 0:
        raise RuntimeError("[HF] No writers satisfy sample count constraints.")

    if len(filtered_writers) < cfg.m:
        print(
            f"[HF] Warning: request m={cfg.m} > filtered writers {len(filtered_writers)}, shrink m."
        )
        cfg.m = len(filtered_writers)

    rng = np.random.default_rng(getattr(cfg, "seed", None))
    selected_writers = rng.choice(filtered_writers, cfg.m, replace=False)

    # Treat each selected writer as one natural federated client.
    all_train_imgs = []
    all_train_labels = []
    split_dict = {}
    cursor = 0
    for cid, w in enumerate(selected_writers):
        inds = writer_to_indices[w]
        for i in inds:
            all_train_imgs.append(images[i])
            all_train_labels.append(labels[i])
        new_inds = list(range(cursor, cursor + len(inds)))
        split_dict[cid] = np.array(new_inds, dtype="int32")
        cursor += len(inds)

        # print(f"[HF] Client {cid} writer={w} samples={len(inds)}")

    try:
        if not hasattr(cfg, "femnist_D"):
            cfg.femnist_D = os.path.join(
                cache_dir, f"client_data_vols_m{cfg.m}_seed{getattr(cfg, 'seed', 'None')}.yaml"
            )
        export_path = cfg.femnist_D
        with open(export_path, "w") as f_yaml:
            client_data_vols = {int(cid): int(len(idxs)) for cid, idxs in split_dict.items()}
            yaml.dump(client_data_vols, f_yaml)
        print(f"[HF] Saved client data volumes to {export_path}")
    except Exception as e:
        print(f"[HF][Warn] Failed to save client data volumes yaml: {e}")

    def pil_to_np(img):
        """Convert PIL Image to NumPy array."""
        if isinstance(img, Image.Image):
            return np.array(img, dtype="uint8")
        return np.array(img, dtype="uint8")

    train_array = np.stack([pil_to_np(im) for im in all_train_imgs])
    train_labels_array = np.array(all_train_labels, dtype=np.int64)

    test_array = np.empty((0, 28, 28), dtype=np.uint8)
    test_labels_array = np.empty((0,), dtype=np.int64)

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset_train = FEMNISTDataset(train_array, train_labels_array, transform_train)
    dataset_test = FEMNISTDataset(test_array, test_labels_array, transform_eval)

    cfg.data_train = len(dataset_train)
    cfg.data_test = len(dataset_test)

    print(f"[HF] Final sizes - Train:{cfg.data_train} Test:{cfg.data_test} (writers {cfg.m})")

    if cfg.data_test > 0:
        _, counts = np.unique(test_labels_array, return_counts=True)
        props = counts / cfg.data_test
        print(
            "[HF] Test label distribution: "
            f"min={np.min(props):.4f}, "
            f"max={np.max(props):.4f}, "
            f"std={np.std(props):.4f}"
        )

    try:
        os.makedirs(processed_dir, exist_ok=True)

        np.save(np_files["train_x"], train_array)
        np.save(np_files["train_y"], train_labels_array)
        np.save(np_files["test_x"], test_array)
        np.save(np_files["test_y"], test_labels_array)

        split_dict_serializable = {int(k): v.tolist() for k, v in split_dict.items()}
        with open(split_path, "w") as fjs:
            json.dump(split_dict_serializable, fjs)

        meta = {
            "signature": signature,
            "params": sig_items,
            "actual_m": cfg.m,
            "data_train": cfg.data_train,
            "data_test": cfg.data_test,
            "num_classes": 62,
            "force_reload_used": force_reload,
        }
        with open(meta_path, "w") as fm:
            yaml.dump(meta, fm)
        print(f"[FEMNIST][Cache] Stored processed data at {processed_dir}")
    except Exception as e:
        print(f"[FEMNIST][Cache][Warn] Failed to save cache: {e}")

    return dataset_train, dataset_test, split_dict
