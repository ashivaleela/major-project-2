"""Model checkpoint saving and loading utilities."""

from __future__ import annotations
import os
from typing import Dict, Any, Optional
import torch


def save_checkpoint(
    agent: Any,
    filepath: str,
    epoch: int,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save agent parameters, optimizer states, and metadata to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    state_dict = {
        "epoch": epoch,
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        "metrics": metrics or {},
        "config": config or getattr(agent, "config", {}),
    }
    torch.save(state_dict, filepath)


def load_checkpoint(
    agent: Any,
    filepath: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load agent weights and optimizer states from a saved checkpoint file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    map_loc = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(filepath, map_location=map_loc)

    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])

    if hasattr(agent, "actor_optimizer") and "actor_optimizer_state_dict" in checkpoint:
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    if hasattr(agent, "critic_optimizer") and "critic_optimizer_state_dict" in checkpoint:
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    return checkpoint

