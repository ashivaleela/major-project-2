"""Observation decomposition, normalization, and tensor encoding."""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """
    Decomposes, sanitizes, and embeds raw observation vectors.
    Partitions features into:
    1. Local satellite state (ID, shell, kinematics, previous allocation)
    2. Terrestrial ground cell states (backlog, visibility, channel gain, elevation)
    3. Shell-level summary statistics (mean queue, active beam ratio, power, RB occupancy, interference)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        net_cfg = config.get("network", {})

        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.num_shells = int(net_cfg.get("num_shells", 2))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_ground_cells = int(net_cfg.get("num_ground_cells", 20))

        self.local_feat_dim = self.num_satellites + self.num_shells + 6 + self.beams_per_satellite
        self.cells_feat_dim = 4 * self.num_ground_cells
        self.shell_sum_dim = 5 * self.num_shells
        self.total_obs_dim = self.local_feat_dim + self.cells_feat_dim + self.shell_sum_dim

        # Neural feature embedding
        hidden_dim = int(config.get("mappo", {}).get("hidden_dim", 128))
        self.feature_net = nn.Sequential(
            nn.Linear(self.total_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Embed normalized observation vector.

        Args:
            obs: Tensor of shape [..., total_obs_dim].

        Returns:
            Embedded representation of shape [..., hidden_dim].
        """
        # Clamp inputs to prevent non-finite numerical anomalies
        sanitized = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return self.feature_net(sanitized)

    def decompose_observation(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Deconstruct a 1D observation vector into named semantic blocks.

        Args:
            obs: 1D NumPy array of length total_obs_dim.

        Returns:
            Dict containing 'local', 'cells', and 'shell_summaries' arrays.
        """
        idx0 = 0
        idx1 = idx0 + self.local_feat_dim
        idx2 = idx1 + self.cells_feat_dim
        idx3 = idx2 + self.shell_sum_dim

        local_part = obs[idx0:idx1]
        cells_part = obs[idx1:idx2]
        shell_part = obs[idx2:idx3]

        return {
            "local": local_part,
            "cells": cells_part.reshape((self.num_ground_cells, 4)),
            "shell_summaries": shell_part.reshape((self.num_shells, 5)),
        }

