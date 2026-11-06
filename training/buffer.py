"""Rollout buffer for multi-agent trajectory storage and batching."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import torch


class RolloutBuffer:
    """
    Storage buffer for multi-agent trajectories across scheduling slots.
    Stores observations, actions, masks, rewards, values, log-probs, and counterfactual credits.
    """

    def __init__(
        self,
        buffer_size: int,
        num_satellites: int,
        obs_dim: int,
        global_state_dim: int,
        beams_per_satellite: int,
        num_cell_actions: int,
        num_rb_actions: int,
    ):
        self.buffer_size = buffer_size
        self.num_satellites = num_satellites
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.beams_per_satellite = beams_per_satellite
        self.num_cell_actions = num_cell_actions
        self.num_rb_actions = num_rb_actions

        self.reset()

    def reset(self) -> None:
        """Clear the buffer."""
        self.obs = np.zeros((self.buffer_size, self.num_satellites, self.obs_dim), dtype=np.float32)
        self.global_states = np.zeros((self.buffer_size, self.global_state_dim), dtype=np.float32)

        self.cell_actions = np.zeros((self.buffer_size, self.num_satellites, self.beams_per_satellite), dtype=np.int64)
        self.rb_actions = np.zeros((self.buffer_size, self.num_satellites, self.beams_per_satellite), dtype=np.int64)
        self.power_actions = np.zeros((self.buffer_size, self.num_satellites, self.beams_per_satellite), dtype=np.float32)

        self.cell_masks = np.zeros((self.buffer_size, self.num_satellites, self.num_cell_actions), dtype=bool)
        self.rb_masks = np.zeros(
            (self.buffer_size, self.num_satellites, self.beams_per_satellite, self.num_rb_actions), dtype=bool
        )

        self.log_probs = np.zeros((self.buffer_size, self.num_satellites), dtype=np.float32)
        self.beam_log_probs = np.zeros((self.buffer_size, self.num_satellites, self.beams_per_satellite), dtype=np.float32)
        self.beam_credits = np.zeros((self.buffer_size, self.num_satellites, self.beams_per_satellite), dtype=np.float32)

        self.rewards = np.zeros((self.buffer_size, self.num_satellites), dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=bool)

        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)

        self.step_idx = 0

    def insert(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state: np.ndarray,
        actions_dict: Dict[str, Any],
        action_arrays_dict: Dict[str, Dict[str, np.ndarray]],
        cell_masks_dict: Dict[str, np.ndarray],
        rb_masks_dict: Dict[str, np.ndarray],
        log_probs_dict: Dict[str, float],
        beam_log_probs_dict: Dict[str, np.ndarray],
        beam_credits_dict: Optional[Dict[str, np.ndarray]],
        rewards_dict: Dict[str, float],
        value: float,
        done: bool,
    ) -> None:
        """Insert a transition step for all satellites."""
        t = self.step_idx

        self.global_states[t] = global_state
        self.values[t] = value
        self.dones[t] = done

        for s_idx in range(self.num_satellites):
            sat_key = f"sat_{s_idx}"
            self.obs[t, s_idx] = obs_dict[sat_key]

            arrs = action_arrays_dict[sat_key]
            self.cell_actions[t, s_idx] = arrs["cell_actions"]
            self.rb_actions[t, s_idx] = arrs["rb_actions"]
            self.power_actions[t, s_idx] = arrs["power_actions"]

            self.cell_masks[t, s_idx] = cell_masks_dict[sat_key]
            self.rb_masks[t, s_idx] = rb_masks_dict[sat_key]

            self.log_probs[t, s_idx] = log_probs_dict[sat_key]
            self.beam_log_probs[t, s_idx] = beam_log_probs_dict[sat_key]

            if beam_credits_dict and sat_key in beam_credits_dict:
                self.beam_credits[t, s_idx] = beam_credits_dict[sat_key]

            self.rewards[t, s_idx] = rewards_dict[sat_key]

        self.step_idx += 1

    def is_full(self) -> bool:
        return self.step_idx >= self.buffer_size

