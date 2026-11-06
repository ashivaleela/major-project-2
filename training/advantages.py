"""Generalized Advantage Estimation (GAE) and discounted return calculations."""

from __future__ import annotations
import numpy as np


def compute_gae_and_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Generalized Advantage Estimation (GAE) and discounted returns.

    Args:
        rewards: 1D array of shape [T] containing slot rewards (mean across satellites or team reward).
        values: 1D array of shape [T] containing critic value estimates V(S_t).
        dones: 1D boolean array of shape [T] indicating terminal transitions.
        last_value: Bootstrapped value V(S_T) for the final transition.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter.

    Returns:
        Tuple of (advantages, returns) of shape [T].
    """
    t_len = len(rewards)
    advantages = np.zeros(t_len, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(t_len)):
        if t == t_len - 1:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = last_value
        else:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize advantages to zero mean and unit variance for stable PPO updates."""
    mean = float(np.mean(advantages))
    std = float(np.std(advantages))
    return (advantages - mean) / max(eps, std)

