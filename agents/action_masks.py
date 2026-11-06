"""Action masking utilities for beam feasibility and intra-satellite RB exclusivity."""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import torch


class ActionMaskManager:
    """Manages dynamic discrete action masks for cell and resource block choices."""

    def __init__(self, num_ground_cells: int = 20, num_rbs: int = 8, beams_per_satellite: int = 2):
        self.num_ground_cells = num_ground_cells
        self.num_rbs = num_rbs
        self.beams_per_satellite = beams_per_satellite

        # Precompute all valid discrete (rb_start, rb_width) pairs
        self.valid_rb_pairs: List[Tuple[int, int]] = []
        for start in range(num_rbs):
            for width in range(1, num_rbs - start + 1):
                self.valid_rb_pairs.append((start, width))
        self.num_rb_actions = len(self.valid_rb_pairs)  # 8*9/2 = 36 valid pairs

    def get_cell_mask(self, visible_cells: List[int]) -> np.ndarray:
        """
        Mask for cell choice: index 0 (inactive) is always valid;
        indices 1..N correspond to ground cells 0..N-1 and are valid if visible.
        """
        mask = np.zeros(self.num_ground_cells + 1, dtype=bool)
        mask[0] = True  # Inactive is always valid
        for c in visible_cells:
            if 0 <= c < self.num_ground_cells:
                mask[c + 1] = True
        return mask

    def get_rb_mask(self, occupied_rbs: set[int]) -> np.ndarray:
        """
        Mask for discrete (rb_start, rb_width) pairs:
        A pair is valid if none of its RBs are in occupied_rbs.
        """
        mask = np.zeros(self.num_rb_actions, dtype=bool)
        for idx, (start, width) in enumerate(self.valid_rb_pairs):
            pair_rbs = set(range(start, start + width))
            if not pair_rbs.intersection(occupied_rbs):
                mask[idx] = True
        # If all pairs collide (no free RBs), enable first pair as fallback
        if not np.any(mask):
            mask[0] = True
        return mask

    def decode_rb_action(self, action_idx: int) -> Tuple[int, int]:
        """Convert discrete RB action index back to (start, width)."""
        idx = int(np.clip(action_idx, 0, self.num_rb_actions - 1))
        return self.valid_rb_pairs[idx]

