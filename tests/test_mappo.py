"""Unit tests for MAPPO algorithm, GAE calculations, and checkpointing."""

import os
import pytest
import numpy as np
import torch
import yaml

from environment.leo_env import MultiShellLEOEnv
from agents.mappo import ShellMeanMAPPO
from training.buffer import RolloutBuffer
from training.advantages import compute_gae_and_returns, normalize_advantages
from training.checkpoint import save_checkpoint, load_checkpoint
from training.rollout import collect_rollout


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["mappo"]["rollout_steps"] = 8
    cfg["mappo"]["mini_batch_size"] = 16
    cfg["mappo"]["ppo_epochs"] = 2
    return cfg


def test_gae_and_advantage_normalization():
    """Verify GAE computation and advantage normalization."""
    rewards = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    values = np.array([2.0, 2.5, 2.0, 3.0], dtype=np.float32)
    dones = np.array([False, False, False, True], dtype=bool)

    adv, returns = compute_gae_and_returns(rewards, values, dones, last_value=1.0, gamma=0.99, gae_lambda=0.95)

    assert len(adv) == 4
    assert len(returns) == 4
    assert np.all(np.isfinite(adv))

    norm_adv = normalize_advantages(adv)
    assert pytest.approx(float(np.mean(norm_adv)), abs=1e-5) == 0.0
    assert pytest.approx(float(np.std(norm_adv)), abs=1e-5) == 1.0


def test_mappo_rollout_and_update(test_config):
    """Test full rollout collection and one PPO update cycle."""
    env = MultiShellLEOEnv(test_config, seed=42)
    agent = ShellMeanMAPPO(test_config)

    buffer = RolloutBuffer(
        buffer_size=8,
        num_satellites=8,
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        beams_per_satellite=2,
        num_cell_actions=agent.actor.num_cell_actions,
        num_rb_actions=agent.mask_mgr.num_rb_actions,
    )

    # Collect rollout
    obs, info, telem = collect_rollout(env, agent, buffer, use_counterfactual_credit=True)
    assert buffer.is_full()
    assert len(buffer.advantages) == 8

    # Form batch for update
    total_samples = 8 * 8
    t_obs = torch.as_tensor(buffer.obs.reshape((total_samples, env.obs_dim)), dtype=torch.float32)
    t_c = torch.as_tensor(buffer.cell_actions.reshape((total_samples, 2)), dtype=torch.int64)
    t_r = torch.as_tensor(buffer.rb_actions.reshape((total_samples, 2)), dtype=torch.int64)
    t_p = torch.as_tensor(buffer.power_actions.reshape((total_samples, 2)), dtype=torch.float32)
    t_cm = torch.as_tensor(buffer.cell_masks.reshape((total_samples, agent.actor.num_cell_actions)), dtype=torch.bool)
    t_rm = torch.as_tensor(buffer.rb_masks.reshape((total_samples, 2, agent.mask_mgr.num_rb_actions)), dtype=torch.bool)
    t_lp = torch.as_tensor(buffer.log_probs.reshape(total_samples), dtype=torch.float32)
    t_adv = torch.as_tensor(np.repeat(normalize_advantages(buffer.advantages), 8), dtype=torch.float32)
    t_gs = torch.as_tensor(np.repeat(buffer.global_states, 8, axis=0), dtype=torch.float32)
    t_ret = torch.as_tensor(np.repeat(buffer.returns, 8).reshape((total_samples, 1)), dtype=torch.float32)
    t_cred = torch.as_tensor(buffer.beam_credits.reshape((total_samples, 2)), dtype=torch.float32)

    loss_dict = agent.update(
        batch_obs=t_obs,
        batch_cell_acts=t_c,
        batch_rb_acts=t_r,
        batch_pwr_acts=t_p,
        batch_cell_masks=t_cm,
        batch_rb_masks=t_rm,
        batch_old_log_probs=t_lp,
        batch_advantages=t_adv,
        batch_global_states=t_gs,
        batch_returns=t_ret,
        batch_beam_credits=t_cred,
    )

    assert "actor_loss" in loss_dict
    assert "critic_loss" in loss_dict
    assert np.isfinite(loss_dict["actor_loss"])
    assert np.isfinite(loss_dict["critic_loss"])


def test_checkpoint_saving_and_loading(test_config, tmp_path):
    """Verify exact parameter preservation across checkpoint save and load."""
    agent1 = ShellMeanMAPPO(test_config)
    agent2 = ShellMeanMAPPO(test_config)

    save_path = str(tmp_path / "model_chk.pt")
    save_checkpoint(agent1, save_path, epoch=5, metrics={"reward": 42.0})

    load_checkpoint(agent2, save_path)

    # Check actor weights match exactly
    for p1, p2 in zip(agent1.actor.parameters(), agent2.actor.parameters()):
        assert torch.all(torch.eq(p1, p2))

