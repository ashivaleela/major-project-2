"""Integration tests for the full MultiShellLEOEnv environment."""

import pytest
import numpy as np
import yaml
from environment.leo_env import MultiShellLEOEnv


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_env_reset(test_config):
    """Test environment initialization and reset."""
    env = MultiShellLEOEnv(test_config, seed=42)
    obs_dict, info = env.reset(seed=42)

    assert len(obs_dict) == 8  # 8 satellites
    for sat_key, obs in obs_dict.items():
        assert obs.shape == (env.obs_dim,)
        assert np.all(np.isfinite(obs))
        assert not np.any(np.isnan(obs))

    assert "queues" in info
    assert len(info["queues"]) == 20
    assert "global_state" in info
    assert info["global_state"].shape == (env.global_state_dim,)


def test_env_step_execution(test_config):
    """Test executing a multi-agent scheduling step with random feasible actions."""
    env = MultiShellLEOEnv(test_config, seed=42)
    obs_dict, _ = env.reset(seed=42)
    masks = env.get_action_masks()

    actions = {}
    for sat_id in range(env.num_satellites):
        sat_key = f"sat_{sat_id}"
        vis = masks[sat_key]["visible_cells"]
        target_cell = vis[0] + 1 if len(vis) > 0 else 0
        actions[sat_key] = {
            "beam_0": {"cell_id": target_cell, "rb_start": 0, "rb_width": 4, "power_share": 0.5},
            "beam_1": {"cell_id": 0, "rb_start": 4, "rb_width": 4, "power_share": 0.0},
        }

    next_obs, rewards, terminated, truncated, info = env.step(actions)

    assert len(next_obs) == 8
    assert len(rewards) == 8
    assert not terminated
    assert not truncated
    assert info["served_traffic_mb"] >= 0.0
    assert "team_reward" in info
    assert np.isfinite(info["team_reward"])


def test_counterfactual_step_non_mutation(test_config):
    """Verify that counterfactual replay does NOT mutate real environment state."""
    env = MultiShellLEOEnv(test_config, seed=42)
    obs_dict, _ = env.reset(seed=42)
    masks = env.get_action_masks()

    actions = {}
    for sat_id in range(env.num_satellites):
        sat_key = f"sat_{sat_id}"
        vis = masks[sat_key]["visible_cells"]
        c = vis[0] + 1 if vis else 0
        actions[sat_key] = {
            "beam_0": {"cell_id": c, "rb_start": 0, "rb_width": 4, "power_share": 0.5},
            "beam_1": {"cell_id": 0, "rb_start": 4, "rb_width": 4, "power_share": 0.0},
        }

    # Record snapshot of current state
    q_before = env.traffic_model.get_queues().copy()
    step_before = env.current_step
    sat0_pos_before = env.satellite_network.satellites[0].position.copy()

    # Evaluate counterfactual step for sat 0, beam 0
    cf_utility = env.evaluate_counterfactual_step(actions, target_sat_id=0, target_beam_id=0)

    # State must be completely identical!
    q_after = env.traffic_model.get_queues()
    step_after = env.current_step
    sat0_pos_after = env.satellite_network.satellites[0].position

    assert np.array_equal(q_before, q_after)
    assert step_before == step_after
    assert np.array_equal(sat0_pos_before, sat0_pos_after)
    assert np.isfinite(cf_utility)

