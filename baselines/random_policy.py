"""Random baseline policy selecting randomly among feasible actions."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


class RandomPolicy:
    """Random baseline scheduler that selects uniformly among feasible cells, RBs, and power."""

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        self.config = config
        self.seed = seed if seed is not None else config.get("experiment", {}).get("seed", 42)
        self.rng = np.random.default_rng(self.seed)

        net_cfg = config.get("network", {})
        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_rbs = int(net_cfg.get("num_resource_blocks", 8))

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        action_masks: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate random actions for all satellites respecting visibility constraints.

        Args:
            observations: Dict mapping 'sat_i' to observation vector.
            action_masks: Dict mapping 'sat_i' to action masks and visible cells list.

        Returns:
            Dict of actions compatible with MultiShellLEOEnv.
        """
        actions = {}

        for sat_id in range(self.num_satellites):
            sat_key = f"sat_{sat_id}"
            vis_cells = action_masks[sat_key]["visible_cells"]

            sat_actions = {}
            # Divide available RBs among beams to avoid intra-satellite conflict
            rbs_per_beam = self.num_rbs // self.beams_per_satellite

            for b_idx in range(self.beams_per_satellite):
                # 20% chance of staying inactive, or if no visible cells
                if not vis_cells or self.rng.random() < 0.2:
                    sat_actions[f"beam_{b_idx}"] = {
                        "cell_id": 0,
                        "rb_start": 0,
                        "rb_width": 0,
                        "power_share": 0.0,
                    }
                else:
                    chosen_cell_idx = int(self.rng.choice(vis_cells))
                    # 1-based cell_id
                    cell_id = chosen_cell_idx + 1

                    rb_start = b_idx * rbs_per_beam
                    rb_width = int(self.rng.integers(1, rbs_per_beam + 1))
                    power_share = float(self.rng.uniform(0.2, 0.5))

                    sat_actions[f"beam_{b_idx}"] = {
                        "cell_id": cell_id,
                        "rb_start": rb_start,
                        "rb_width": rb_width,
                        "power_share": power_share,
                    }

            actions[sat_key] = sat_actions

        return actions

