"""Unit tests for AutoregressiveActor, action masking, and gradient flow."""

import pytest
import numpy as np
import torch
import yaml
from agents.actor import AutoregressiveActor
from agents.critic import CentralizedCritic
from agents.action_masks import ActionMaskManager
from environment.leo_env import MultiShellLEOEnv


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_actor_sampling_and_mask_enforcement(test_config):
    """Verify actor samples only feasible choices respecting masks."""
    actor = AutoregressiveActor(test_config)
    env = MultiShellLEOEnv(test_config, seed=42)
    obs_dict, _ = env.reset(seed=42)
    masks_dict = env.get_action_masks()

    obs0 = torch.as_tensor(obs_dict["sat_0"], dtype=torch.float32)
    mask0 = masks_dict["sat_0"]
    vis_cells = set(mask0["visible_cells"])

    # Sample actions
    act_dict, tot_lp, b_lps = actor.forward_sample(obs0, mask0, deterministic=False)

    assert "beam_0" in act_dict
    assert "beam_1" in act_dict
    assert torch.isfinite(tot_lp)
    assert len(b_lps) == 2

    for b in range(2):
        b_act = act_dict[f"beam_{b}"]
        c_choice = b_act["cell_id"]
        # Must be either 0 (inactive) or one of the visible cell indices + 1
        if c_choice > 0:
            assert (c_choice - 1) in vis_cells
        assert 0.0 <= b_act["power_share"] <= 1.0


def test_actor_critic_gradient_flow(test_config):
    """Verify backward gradient computation through Actor and Critic."""
    actor = AutoregressiveActor(test_config)
    critic = CentralizedCritic(test_config)

    batch_size = 4
    obs_dim = actor.obs_dim
    global_dim = critic.global_state_dim

    obs = torch.randn(batch_size, obs_dim, requires_grad=False)
    global_s = torch.randn(batch_size, global_dim, requires_grad=False)

    cell_acts = torch.randint(0, 5, (batch_size, 2))
    rb_acts = torch.randint(0, 10, (batch_size, 2))
    pwr_acts = torch.rand(batch_size, 2)
    cell_masks = torch.ones(batch_size, actor.num_cell_actions, dtype=torch.bool)
    rb_masks = torch.ones(batch_size, 2, actor.num_rb_actions, dtype=torch.bool)

    tot_lp, b_lps, entropy = actor.evaluate_actions(
        obs, cell_acts, rb_acts, pwr_acts, cell_masks, rb_masks
    )

    actor_loss = -tot_lp.mean() - 0.01 * entropy.mean()
    actor_loss.backward()

    # Verify gradients exist and are finite
    for p in actor.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))

    # Critic gradient check
    val = critic(global_s)
    critic_loss = (val - 1.0).pow(2).mean()
    critic_loss.backward()

    for p in critic.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))

