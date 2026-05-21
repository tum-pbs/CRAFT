"""Main entry point for CRAFT federated learning training."""

from fed.client import Clients
from fed.dataset import init_dataset
from fed.federate import FL_train
from fed.server import Server
from utils.args import args_parser
from utils.utils import load_cfg, select_device, set_seed


def main(cfg):
    """Set up data, server, clients, and run CRAFT training."""
    set_seed(cfg.seed)  # Set the random seed for reproducibility.
    select_device(cfg)  # Select the training device.
    dataset = init_dataset(cfg)  # Initialize the configured dataset.
    server = Server(cfg)  # Initialize the federated server.
    clients = Clients(cfg, dataset)  # Initialize all federated clients.
    FL_train(cfg, server, clients, dataset)  # Run the federated training loop.


if __name__ == "__main__":
    arg = args_parser()  # Parse command-line arguments.
    cfg = load_cfg(arg)  # Load the selected configuration file.
    main(cfg)
