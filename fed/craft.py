"""Server-side CRAFT aggregation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from utils.utils import apply_model_state

StateDict = Dict[str, torch.Tensor]


class CRAFT:
    """CRAFT aggregation with layerwise conflict mitigation."""

    def __init__(self, cfg, model: torch.nn.Module):
        """Initialize CRAFT state reused across communication rounds."""
        self.cfg = cfg
        self.reference_state = copy.deepcopy(model.state_dict())
        self.server_step_size = float(getattr(cfg, "server_lr", 1.0))
        self.state_specs: List[_StateTensorSpec] = []
        self.flat_update_size = 0
        self.previous_aggregate_update: Optional[torch.Tensor] = None
        self.layer_group_slices = self._build_layer_group_slices(model)

    def aggregate(self, server, res_clients: dict) -> None:
        """Aggregate selected client states and update the server model."""
        selected_client_ids = self._resolve_selected_client_ids(server.cfg.m, res_clients)
        current_state = server.state
        client_state_by_id = res_clients["models"]
        device = next(server.model.parameters()).device

        client_update_matrix = self._stack_client_updates(
            client_state_by_id=client_state_by_id,
            current_state=current_state,
            selected_client_ids=selected_client_ids,
            device=device,
        )
        selected_client_weights = self._make_selected_client_weights(
            all_client_weights=server.cfg.rho,
            selected_client_ids=selected_client_ids,
            like=client_update_matrix,
        )

        aggregate_update = self._compute_layerwise_craft_update(
            client_update_matrix=client_update_matrix,
            selected_client_weights=selected_client_weights,
        )
        next_state = self._apply_server_update(current_state, aggregate_update)

        apply_model_state(server.model, next_state)
        server.state = server.model.state_dict()

    def _compute_layerwise_craft_update(
        self,
        client_update_matrix: torch.Tensor,
        selected_client_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the flattened CRAFT update by solving each layer group."""
        layer_updates = []
        for layer_slice in self.layer_group_slices:
            previous_layer_update = (
                self.previous_aggregate_update[layer_slice]
                if self.previous_aggregate_update is not None
                else None
            )
            layer_update = self._solve_layer_group_update(
                layer_client_updates=client_update_matrix[:, layer_slice],
                selected_client_weights=selected_client_weights,
                previous_layer_update=previous_layer_update,
            )
            layer_updates.append(layer_update)

        aggregate_update = torch.cat(layer_updates)
        self.previous_aggregate_update = aggregate_update.detach().clone()
        return aggregate_update

    def _solve_layer_group_update(
        self,
        layer_client_updates: torch.Tensor,
        selected_client_weights: torch.Tensor,
        previous_layer_update: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Solve the CRAFT projection problem for one layer group.

        Variable names map to the paper notation as follows:
            - layer_client_updates: holds g_i^{t,q}.
            - update_directions: holds row-normalized local updates U(g_i^{t,q}).
            - reference_direction: holds hat{g}_t^q.
            - projection_residual: holds rho_t - U_t^q hat{g}_t^q.
        """
        if layer_client_updates.ndim != 2:
            raise ValueError("Layer client updates must be a 2-dimensional tensor.")

        layer_width = layer_client_updates.shape[1]
        client_update_norms = torch.linalg.vector_norm(layer_client_updates, dim=1)
        usable_client_mask = client_update_norms > 0
        if not torch.any(usable_client_mask):
            return layer_client_updates.new_zeros(layer_width)

        usable_updates = layer_client_updates[usable_client_mask]
        if usable_updates.shape[0] == 1:
            return usable_updates[0]

        usable_update_norms = client_update_norms[usable_client_mask]
        usable_client_weights = self._normalize_client_weights(
            selected_client_weights[usable_client_mask]
        )

        update_directions = torch.nan_to_num(
            usable_updates / usable_update_norms.unsqueeze(1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        reference_direction = self._normalized_reference_direction(
            previous_layer_update=previous_layer_update,
            layer_width=layer_width,
            like=layer_client_updates,
        )

        projection_residual = usable_client_weights - update_directions @ reference_direction
        layer_update = (
            reference_direction + torch.linalg.pinv(update_directions) @ projection_residual
        )

        if torch.linalg.vector_norm(layer_update) <= 0:
            return layer_client_updates.new_zeros(layer_width)
        return layer_update

    def _normalized_reference_direction(
        self,
        previous_layer_update: Optional[torch.Tensor],
        layer_width: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Return the previous layer update as a unit vector, or zero in round 0."""
        if previous_layer_update is None:
            return like.new_zeros(layer_width)

        previous_layer_update = previous_layer_update.to(device=like.device, dtype=like.dtype)
        if previous_layer_update.numel() != layer_width:
            raise ValueError(
                "Previous layer update has incompatible width: "
                f"{previous_layer_update.numel()} != {layer_width}"
            )

        previous_update_norm = torch.linalg.vector_norm(previous_layer_update)
        if previous_update_norm > 0:
            return previous_layer_update / previous_update_norm
        return like.new_zeros(layer_width)

    def _apply_server_update(
        self,
        current_state: StateDict,
        aggregate_update: torch.Tensor,
    ) -> StateDict:
        """Restore and apply the flattened server update to the model state."""
        if aggregate_update.numel() != self.flat_update_size:
            raise ValueError(
                "Aggregate update has incompatible size: "
                f"{aggregate_update.numel()} != {self.flat_update_size}"
            )

        next_state: StateDict = {
            name: tensor.detach().clone() for name, tensor in current_state.items()
        }
        cursor = 0
        for spec in self.state_specs:
            update_block = aggregate_update[cursor : cursor + spec.numel].view(spec.shape)
            cursor += spec.numel

            target_tensor = current_state[spec.name]
            scaled_update = (self.server_step_size * update_block).to(target_tensor.device)
            next_state[spec.name] = target_tensor - scaled_update
        return next_state

    @staticmethod
    def _resolve_selected_client_ids(num_clients: int, res_clients: dict) -> List[int]:
        """Return selected client IDs; use all clients only when the key is absent."""
        raw_client_ids = res_clients.get("selected_clients")
        if raw_client_ids is None:
            raw_client_ids = range(num_clients)

        selected_client_ids = [int(client_id) for client_id in raw_client_ids]
        if not selected_client_ids:
            raise ValueError("CRAFT aggregation requires at least one selected client.")
        return selected_client_ids

    def _stack_client_updates(
        self,
        client_state_by_id: Dict[int, StateDict],
        current_state: StateDict,
        selected_client_ids: Iterable[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build one flattened update row per selected client.

        For client i, the row is:
            vec(theta_t - theta_i^{t,K_i})
        """
        selected_client_ids = list(selected_client_ids)
        missing_client_ids = [
            client_id for client_id in selected_client_ids if client_id not in client_state_by_id
        ]
        if missing_client_ids:
            raise KeyError(f"Missing local model states for clients: {missing_client_ids}")

        return torch.stack(
            [
                self._client_update_vector(client_state_by_id[client_id], current_state, device)
                for client_id in selected_client_ids
            ],
            dim=0,
        )

    def _client_update_vector(
        self,
        client_state: StateDict,
        current_state: StateDict,
        device: torch.device,
    ) -> torch.Tensor:
        """Flatten a client's delta from the current global state."""
        chunks = []
        for spec in self.state_specs:
            global_tensor = current_state[spec.name].to(device)
            client_tensor = client_state[spec.name].to(device)
            chunks.append((global_tensor - client_tensor).reshape(-1))
        return torch.cat(chunks)

    def _make_selected_client_weights(
        self,
        all_client_weights: Sequence[float],
        selected_client_ids: Sequence[int],
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Build normalized data-ratio weights for selected clients."""
        raw_weights = like.new_tensor(
            [float(all_client_weights[client_id]) for client_id in selected_client_ids]
        )
        return self._normalize_client_weights(raw_weights)

    @staticmethod
    def _normalize_client_weights(weights: torch.Tensor) -> torch.Tensor:
        """Normalize non-negative weights, using uniform weights if all are zero."""
        if weights.ndim != 1:
            raise ValueError("Client weights must be a 1-dimensional tensor.")
        if weights.numel() == 0:
            raise ValueError("Client weights cannot be empty.")
        if torch.any(weights < 0):
            raise ValueError("Client weights must be non-negative.")

        total_weight = weights.sum()
        if total_weight.item() <= 0:
            return torch.full_like(weights, 1.0 / weights.numel())
        return weights / total_weight

    def _build_layer_group_slices(self, model: torch.nn.Module) -> List[slice]:
        """Build contiguous flattened slices for CRAFT layer groups."""
        self._refresh_state_specs()
        normalization_owner_by_module = self._map_norm_to_previous_conv(model)
        layer_group_slices = self._build_contiguous_layer_slices(normalization_owner_by_module)
        if not layer_group_slices:
            raise ValueError("CRAFT could not build any layer group slices.")
        return layer_group_slices

    def _refresh_state_specs(self) -> None:
        """Refresh state-tensor metadata in flattening order."""
        self.state_specs = []
        self.flat_update_size = 0
        for name, tensor in self.reference_state.items():
            spec = _StateTensorSpec(
                name=name,
                shape=tuple(tensor.shape),
                numel=tensor.numel(),
            )
            self.state_specs.append(spec)
            self.flat_update_size += spec.numel

        if not self.state_specs:
            raise ValueError("CRAFT requires at least one state tensor.")

    def _build_contiguous_layer_slices(
        self,
        normalization_owner_by_module: Dict[str, str],
    ) -> List[slice]:
        """Group adjacent flattened tensors by their CRAFT layer-group key."""
        layer_group_slices: List[slice] = []
        offset = 0
        current_group_start = 0
        current_group_key: Optional[str] = None

        for spec in self.state_specs:
            next_group_key = self._layer_group_key(spec.name, normalization_owner_by_module)
            if current_group_key is None:
                current_group_key = next_group_key
                current_group_start = offset
            elif next_group_key != current_group_key:
                layer_group_slices.append(slice(current_group_start, offset))
                current_group_key = next_group_key
                current_group_start = offset
            offset += spec.numel

        if current_group_key is not None:
            layer_group_slices.append(slice(current_group_start, offset))
        return layer_group_slices

    def _layer_group_key(
        self,
        state_name: str,
        normalization_owner_by_module: Dict[str, str],
    ) -> str:
        """Return the layer-group key for a flattened state tensor."""
        if state_name.endswith((".weight", ".bias")):
            module_key = state_name.rsplit(".", 1)[0]
            return normalization_owner_by_module.get(module_key, module_key)
        return state_name

    def _map_norm_to_previous_conv(self, model: torch.nn.Module) -> Dict[str, str]:
        """Map normalization modules to the preceding convolution in the same block."""
        norm_types = (torch.nn.GroupNorm, torch.nn.LayerNorm)
        normalization_owner_by_module: Dict[str, str] = {}

        def walk(module: torch.nn.Module, prefix: str = "") -> None:
            previous_conv_name: Optional[str] = None
            for child_name, child in module.named_children():
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, torch.nn.modules.conv._ConvNd):
                    previous_conv_name = full_name
                elif isinstance(child, norm_types) and previous_conv_name is not None:
                    normalization_owner_by_module[full_name] = previous_conv_name
                walk(child, full_name)

        walk(model)
        return normalization_owner_by_module


@dataclass(frozen=True)
class _StateTensorSpec:
    """Metadata for flattening and restoring one state tensor."""

    name: str
    shape: Tuple[int, ...]
    numel: int
