"""Rule-based conflict-aware scheduler using shell partitioning and collision avoidance."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


class RuleBasedPolicy:
    """
    Conflict-aware rule-based scheduler.
    Key heuristics:
    1. Shell Specialization: Shell 0 (Low LEO) serves hotspot and high-urgency cells;
       Shell 1 (High LEO) serves regular demand.
    2. Inter-Shell Frequency Orthogonalization:
       Shell 0 utilizes the lower RB block (e.g. RBs 0..3), while Shell 1 utilizes
       the upper RB block (e.g. RBs 4..7), eliminating cross-shell co-channel interference.
    3. Intra-Shell Spatial Collision Avoidance:
       Satellites only request cells where they have superior channel conditions,
       avoiding wasteful multi-satellite collisions on low-backlog cells.
    """

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        self.config = config
        self.seed = seed if seed is not None else config.get("experiment", {}).get("seed", 42)
        self.rng = np.random.default_rng(self.seed)

        net_cfg = config.get("network", {})
        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.num_shells = int(net_cfg.get("num_shells", 2))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_rbs = int(net_cfg.get("num_resource_blocks", 8))
        self.num_cells = int(net_cfg.get("num_ground_cells", 20))

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
        Generate coordinated rule-based actions for all satellites.

        Args:
            observations: Dict of per-satellite observations.
            action_masks: Dict of action masks.
            env: MultiShellLEOEnv instance for global state access.

        Returns:
            Dict of actions.
        """
        actions = {}

        if env is not None:
            queues = env.traffic_model.get_queues()
            hotspot_flags = env.traffic_model.get_hotspot_flags()
            sats = env.satellite_network.satellites
        else:
            queues = np.ones(self.num_cells, dtype=np.float32) * 20.0
            hotspot_flags = np.zeros(self.num_cells, dtype=bool)
            sats = None

        # Track allocated cells within scheduling round to minimize collision
        allocated_cell_counts: Dict[int, int] = {c: 0 for c in range(self.num_cells)}

        # Process satellites shell by shell
        for sat_id in range(self.num_satellites):
            sat_key = f"sat_{sat_id}"
            vis_cells = action_masks[sat_key]["visible_cells"]
            sat = sats[sat_id] if sats is not None else None
            shell_id = sat.shell_id if sat is not None else (sat_id // 4)

            # 1. Frequency partitioning by shell:
            # Shell 0 gets RBs [0..3], Shell 1 gets RBs [4..7]
            half_rbs = self.num_rbs // 2  # 4
            base_rb_start = shell_id * half_rbs
            rbs_per_beam = half_rbs // self.beams_per_satellite  # 2

            # 2. Priority scoring of visible cells
            scored_candidates = []
            for c_idx in vis_cells:
                q_val = float(queues[c_idx])
                if q_val <= 0.0:
                    continue

                is_hotspot = bool(hotspot_flags[c_idx])
                h_val = float(sat.channel_gains[c_idx]) if sat is not None else 1e-11

                # Priority: Shell 0 prefers hotspots, Shell 1 prefers non-hotspots
                shell_fit = 2.0 if (shell_id == 0 and is_hotspot) or (shell_id == 1 and not is_hotspot) else 1.0

                # Collision avoidance penalty: penalize cell if already targeted by another satellite
                already_assigned = allocated_cell_counts[c_idx]
                collision_penalty = 1.0 / (1.0 + already_assigned * 2.0)

                score = (q_val / 20.0) * shell_fit * collision_penalty * (h_val * 1e11)
                scored_candidates.append((score, c_idx, q_val))

            scored_candidates.sort(key=lambda x: -x[0])

            sat_actions = {}
            for b_idx in range(self.beams_per_satellite):
                if b_idx < len(scored_candidates):
                    _, chosen_c, q_backlog = scored_candidates[b_idx]
                    allocated_cell_counts[chosen_c] += 1
                    cell_id = chosen_c + 1

                    rb_start = base_rb_start + b_idx * rbs_per_beam
                    rb_width = rbs_per_beam
                    power_share = float(np.clip(0.35 + 0.15 * (q_backlog / 50.0), 0.25, 0.45))

                    sat_actions[f"beam_{b_idx}"] = {
                        "cell_id": cell_id,
                        "rb_start": rb_start,
                        "rb_width": rb_width,
                        "power_share": power_share,
                    }
                else:
                    sat_actions[f"beam_{b_idx}"] = {
                        "cell_id": 0,
                        "rb_start": 0,
                        "rb_width": 0,
                        "power_share": 0.0,
                    }

            actions[sat_key] = sat_actions

        return actions

