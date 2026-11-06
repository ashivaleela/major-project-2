"""Unit tests for observation encoder, dimensions, and sanitation."""

import pytest
import numpy as np
import torch
import yaml
from agents.observation_encoder import ObservationEncoder
from environment.leo_env import MultiShellLEOEnv


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_observation_dimensions_and_finite_values(test_config):
    """Verify observation encoder dimensions and numerical stability."""
    env = MultiShellLEOEnv(test_config, seed=42)
    obs_dict, info = env.reset(seed=42)

    encoder = ObservationEncoder(test_config)

    for sat_key, obs in obs_dict.items():
        assert len(obs) == encoder.total_obs_dim
        assert np.all(np.isfinite(obs))
        assert not np.any(np.isnan(obs))

        # Check decomposition
        parts = encoder.decompose_observation(obs)
        assert len(parts["local"]) == encoder.local_feat_dim
        assert parts["cells"].shape == (20, 4)
        assert parts["shell_summaries"].shape == (2, 5)

        # Forward pass through embedding net
        t_obs = torch.as_tensor(obs, dtype=torch.float32)
        embedded = encoder(t_obs)
        assert embedded.shape == (128,)
        assert torch.all(torch.isfinite(embedded))


def test_nan_sanitation(test_config):
    """Verify that corrupt non-finite inputs are safely sanitized."""
    encoder = ObservationEncoder(test_config)
    corrupt_obs = torch.full((encoder.total_obs_dim,), float("nan"))
    corrupt_obs[5] = float("inf")
    corrupt_obs[10] = -float("inf")

    out = encoder(corrupt_obs)
    assert torch.all(torch.isfinite(out))

