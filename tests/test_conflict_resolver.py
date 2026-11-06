"""Unit tests for deterministic conflict resolution."""

import pytest
import numpy as np
import yaml
from environment.conflict_resolver import ConflictResolver, ServiceRequest


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_empty_queue_retains_zero_requests(test_config):
    """Empty queues (backlog = 0) must retain zero requests."""
    resolver = ConflictResolver(test_config)
    queue_backlogs = np.zeros(20, dtype=np.float32)
    visibility_matrix = np.ones((8, 20), dtype=bool)

    req = ServiceRequest(
        sat_id=0,
        shell_id=0,
        beam_id=0,
        cell_id=5,  # cell index 4
        power_share=0.5,
        rb_start=0,
        rb_width=2,
        channel_gain=1e-11,
        propagation_delay_sec=0.003,
    )

    outcome = resolver.resolve_conflicts([req], queue_backlogs, visibility_matrix)
    assert len(outcome.retained_requests) == 0
    assert len(outcome.rejected_requests) == 1
    assert "zero pending traffic queue" in outcome.rejection_reasons[(0, 0)]


def test_retention_limit_not_exceeded(test_config):
    """Retained count must never exceed M_c(t)."""
    resolver = ConflictResolver(test_config)
    # 5 satellites compete for cell 2 (index 1) which has 8 Mb queue (limit = ceil(8/12) = 1)
    queue_backlogs = np.zeros(20, dtype=np.float32)
    queue_backlogs[1] = 8.0  # Limit is 1 beam
    visibility_matrix = np.ones((8, 20), dtype=bool)

    requests = []
    for s_id in range(5):
        requests.append(
            ServiceRequest(
                sat_id=s_id,
                shell_id=0,
                beam_id=0,
                cell_id=2,
                power_share=0.4,
                rb_start=0,
                rb_width=2,
                channel_gain=1e-11 * (1.0 + s_id * 0.1),
                propagation_delay_sec=0.003,
            )
        )

    outcome = resolver.resolve_conflicts(requests, queue_backlogs, visibility_matrix)
    assert len(outcome.retained_requests) == 1
    assert len(outcome.rejected_requests) == 4
    # Highest channel gain (sat 4) must be retained
    assert outcome.retained_requests[0].sat_id == 4


def test_deterministic_ranking_and_tie_breaking(test_config):
    """Identical scores must break ties deterministically by sat_id asc, beam_id asc."""
    resolver = ConflictResolver(test_config)
    queue_backlogs = np.zeros(20, dtype=np.float32)
    queue_backlogs[3] = 100.0  # High backlog, limit = M_c_max = 2
    visibility_matrix = np.ones((8, 20), dtype=bool)

    # 4 satellites with identical channel gain, power, and delay
    requests = []
    for s_id in [3, 1, 0, 2]:  # Intentionally scrambled input order
        requests.append(
            ServiceRequest(
                sat_id=s_id,
                shell_id=0,
                beam_id=0,
                cell_id=4,  # cell 3
                power_share=0.5,
                rb_start=0,
                rb_width=2,
                channel_gain=2e-11,
                propagation_delay_sec=0.003,
            )
        )

    outcome = resolver.resolve_conflicts(requests, queue_backlogs, visibility_matrix)
    assert len(outcome.retained_requests) == 2
    assert len(outcome.rejected_requests) == 2

    # Deterministic tie-breaking must pick sat 0 and sat 1
    retained_sats = [r.sat_id for r in outcome.retained_requests]
    assert retained_sats == [0, 1]


def test_total_request_accounting(test_config):
    """All submitted requests must be strictly partitioned into retained or rejected."""
    resolver = ConflictResolver(test_config)
    queue_backlogs = np.array([5.0] * 20, dtype=np.float32)
    visibility_matrix = np.random.default_rng(42).choice([True, False], size=(8, 20), p=[0.7, 0.3])

    requests = []
    for s_id in range(8):
        for b_id in range(2):
            target_cell = (s_id * 2 + b_id) % 20 + 1
            requests.append(
                ServiceRequest(
                    sat_id=s_id,
                    shell_id=s_id // 4,
                    beam_id=b_id,
                    cell_id=target_cell,
                    power_share=0.5,
                    rb_start=b_id * 4,
                    rb_width=4,
                    channel_gain=1.5e-11,
                    propagation_delay_sec=0.003,
                )
            )

    outcome = resolver.resolve_conflicts(requests, queue_backlogs, visibility_matrix)
    assert len(outcome.retained_requests) + len(outcome.rejected_requests) == len(requests)

