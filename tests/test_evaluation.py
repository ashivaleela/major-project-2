"""Unit tests for baseline policies and evaluation metrics."""

import pytest
import yaml
import numpy as np
from environment.leo_env import MultiShellLEOEnv
from baselines.random_policy import RandomPolicy
from baselines.greedy_policy import GreedyPolicy
from baselines.rule_based_policy import RuleBasedPolicy
from evaluation.evaluate import run_episode, evaluate_policy


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # Fast test episode
    cfg["network"]["episode_length"] = 5
    return cfg


def test_random_policy_execution(test_config):
    """Test running an episode with RandomPolicy."""
    env = MultiShellLEOEnv(test_config, seed=42)
    policy = RandomPolicy(test_config, seed=42)
    summary = run_episode(env, policy, seed=42)

    assert len(summary.slot_metrics) == 5
    assert summary.total_served_mb >= 0.0
    assert 0.0 <= summary.mean_jains_fairness <= 1.0


def test_greedy_policy_execution(test_config):
    """Test running an episode with GreedyPolicy."""
    env = MultiShellLEOEnv(test_config, seed=42)
    policy = GreedyPolicy(test_config, seed=42)
    summary = run_episode(env, policy, seed=42)

    assert len(summary.slot_metrics) == 5
    assert summary.total_served_mb > 0.0


def test_rule_based_policy_execution(test_config):
    """Test running an episode with RuleBasedPolicy."""
    env = MultiShellLEOEnv(test_config, seed=42)
    policy = RuleBasedPolicy(test_config, seed=42)
    summary = run_episode(env, policy, seed=42)

    assert len(summary.slot_metrics) == 5
    assert summary.total_served_mb > 0.0
    # Rule-based policy partitions RBs, so interference should be low
    assert summary.mean_retention_rate > 50.0


def test_multi_seed_evaluation(test_config):
    """Test multi-seed evaluation aggregator."""
    env = MultiShellLEOEnv(test_config, seed=42)
    policy = RuleBasedPolicy(test_config, seed=42)
    res = evaluate_policy(env, policy, "rule_based", seeds=[42, 101])

    assert len(res["dataframe"]) == 2
    assert "total_served_mb" in res["means"]
    assert "mean_throughput_mbps" in res["means"]

