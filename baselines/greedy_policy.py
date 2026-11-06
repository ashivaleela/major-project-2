"""Greedy baseline scheduler prioritizing queue backlog and channel gain."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


class GreedyPolicy:
    """
    Greedy baseline scheduler.
    Ranks visible cells by a composite demand-channel score:
        Score = w_q * (Q_c / Q_max) + w_h * (h_{i,c} / h_mean)
    Allocates primary beams to highest-demand cells with proportional power.
    """

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        self.config = config
        self.seed = seed if seed is not None else config.get("experiment", {}).get("seed", 42)
        self.rng = np.random.default_rng(self.seed)

        net_cfg = config.get("network", {})
        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_rbs = int(net_cfg.get("num_resource_blocks", 8))
        self.num_cells = int(net_cfg.get("num_ground_cells", 20))

        self.w_q = 1.5
        self.w_h = 1.0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        action_masks: Dict[str, Dict[str, Any]],
        env: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate greedy actions for each satellite.

        Args:
            observations: Dict of per-satellite observation vectors.
            action_masks: Action masks containing visible cell indices.
            env: MultiShellLEOEnv instance (to read exact current queues and channel gains).

        Returns:
            Dict of actions compatible with MultiShellLEOEnv.
        """
        actions = {}
        rbs_per_beam = self.num_rbs // self.beams_per_satellite

        if env is not None:
            queues = env.traffic_model.get_queues()
            sats = env.satellite_network.satellites
        else:
            # Fallback if env reference not passed
            queues = np.ones(self.num_cells, dtype=np.float32) * 20.0
            sats = None

        for sat_id in range(self.num_satellites):
            sat_key = f"sat_{sat_id}"
            vis_cells = action_masks[sat_key]["visible_cells"]
            sat = sats[sat_id] if sats is not None else None

            # Score visible cells
            scored_cells = []
            for c_idx in vis_cells:
                q_val = float(queues[c_idx])
                if q_val <= 0.0:
                    continue
                h_val = float(sat.channel_gains[c_idx]) if sat is not None else 1e-11
                # Normalization
                score = self.w_q * (q_val / 100.0) + self.w_h * (h_val * 1e11)
                scored_cells.append((score, c_idx, q_val))

            # Sort descending by greedy score
            scored_cells.sort(key=lambda x: -x[0])

            sat_actions = {}
            for b_idx in range(self.beams_per_satellite):
                if b_idx < len(scored_cells):
                    _, chosen_c, q_backlog = scored_cells[b_idx]
                    cell_id = chosen_c + 1
                    rb_start = b_idx * rbs_per_beam
                    rb_width = rbs_per_beam
                    # Power share: higher for higher backlog
                    power_share = float(np.clip(0.3 + 0.2 * (q_backlog / 50.0), 0.2, 0.5))

                    sat_actions[f"beam_{b_idx}"] = {
                        "cell_id": cell_id,
                        "rb_start": rb_start,
                        "rb_width": rb_width,
                        "power_share": power_share,
                    }
                else:
                    # Inactive
                    sat_actions[f"beam_{b_idx}"] = {
                        "cell_id": 0,
                        "rb_start": 0,
                        "rb_width": 0,
                        "power_share": 0.0,
                    }

            actions[sat_key] = sat_actions

        return actions

