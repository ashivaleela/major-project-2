"""Centralized Critic network estimating state-value V(S) for multi-agent training."""

from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn


class CentralizedCritic(nn.Module):
    """
    Centralized Critic network.
    Receives global network state S(t) (concatenation of all satellites' local states,
    global ground cell queue vector, and shell-level summaries) and predicts
    the expected cumulative team return V(S(t)).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        net_cfg = config.get("network", {})
        mappo_cfg = config.get("mappo", {})

        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.num_shells = int(net_cfg.get("num_shells", 2))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_ground_cells = int(net_cfg.get("num_ground_cells", 20))

        self.local_feat_dim = self.num_satellites + self.num_shells + 6 + self.beams_per_satellite
        self.global_state_dim = (
            (self.num_satellites * self.local_feat_dim)
            + (2 * self.num_ground_cells)
            + (5 * self.num_shells)
        )

        self.hidden_dim = int(mappo_cfg.get("hidden_dim", 128))

        self.v_net = nn.Sequential(
            nn.Linear(self.global_state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value V(S).

        Args:
            global_state: Tensor of shape [..., global_state_dim].

        Returns:
            Scalar value estimate tensor of shape [..., 1].
        """
        sanitized = torch.nan_to_num(global_state, nan=0.0, posinf=10.0, neginf=-10.0)
        return self.v_net(sanitized)

