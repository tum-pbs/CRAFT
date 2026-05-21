"""Command-line argument parser for federated learning experiments."""

import argparse
import os


def args_parser(cfg_file="config.yaml"):
    """Parse command-line arguments for federated learning experiments."""
    args = argparse.ArgumentParser()

    path_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_file)

    args.add_argument("--cfg", default=path_cfg, type=str, help="path to the config")
    args.add_argument("--alg", type=str, help="FL algorithm")
    args.add_argument("--server_lr", type=float, help="server-side CRAFT update scale")
    args.add_argument("--bs", type=int, help="size of local mini-batches")
    args.add_argument("--E", type=int, help="number of epochs")
    args.add_argument("--T", type=int, help="number of training rounds")
    args.add_argument("--lr", type=float, help="learning rate")
    args.add_argument("--seed", type=int, help="seed")
    args.add_argument("--dataset", type=str, help="name of the dataset")
    args.add_argument("--m", type=int, help="number of users")
    args.add_argument("--frac", type=float, help="active clients per round")
    args.add_argument("--partition", type=str, help="data partition: iid or dir (dirichlet)")
    args.add_argument("--dir_alpha", type=float, help="Dirichlet alpha for non-iid splits")
    args.add_argument("--balance", type=str2bool, help="balance dataset separation")
    args.add_argument("--model", type=str, help="name of the model")
    args.add_argument("--tag", type=str, help="tag for saving")
    args.add_argument("--debug", type=str2bool, help="debug mode")
    args.add_argument("--dir", type=str, default="cfgs", help="Directory for repeat cfg files")
    args.add_argument("--jobid", type=int, default=0, help="Cluster job ID")

    args = args.parse_args()
    return args


def str2bool(str):
    """Convert a string representation to a boolean value."""
    if isinstance(str, bool):
        return str

    if str.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    elif str.lower() in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
