"""Federated learning server module."""

import numpy as np

from fed.craft import CRAFT
from fed.nets import init_model


class Server:
    """Server-side operations for federated learning."""

    def __init__(self, cfg):
        """Initialize the server with the global model."""
        self.cfg = cfg
        self.model = init_model(cfg)
        self.state = self.model.state_dict()
        self.algorithm = CRAFT(cfg, self.model)  # Use CRAFT as the aggregation algorithm.
        self.active_clients = None

    def select_clients(self, frac):
        """Randomly select clients to participate in the current round."""
        # frac <= 1 is a participation ratio; frac > 1 is treated as an absolute count.
        num = int(np.ceil(frac if frac > 1 else frac * self.cfg.m))
        # Always select at least one client and at most all clients.
        num = max(1, min(num, self.cfg.m))
        self.active_clients = np.random.choice(range(self.cfg.m), num, replace=False)
        return self.active_clients

    def aggregate(self, res_clients: dict):
        """Delegate model aggregation to the attached algorithm."""
        self.algorithm.aggregate(self, res_clients)
