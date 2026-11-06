"""Unit tests for ground cell traffic queues and arrival processes."""

import pytest
import numpy as np
import yaml
from environment.traffic_model import TrafficModel, GroundCell


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_ground_cell_queue_update():
    """Verify Q_c(t+1) = max(0, Q_c(t) - x_c(t)) + A_c(t)."""
    cell = GroundCell(cell_id=0, position=np.zeros(3), max_capacity=100.0)
    cell.reset(initial_backlog=20.0)
    assert cell.queue_backlog == 20.0

    # Service 8.0 Mb, arrival 5.0 Mb -> 20 - 8 + 5 = 17.0 Mb
    eff_s, dropped = cell.step_queue(served_traffic=8.0, arrival_traffic=5.0)
    assert eff_s == 8.0
    assert dropped == 0.0
    assert pytest.approx(cell.queue_backlog, 1e-5) == 17.0


def test_service_capped_at_queue():
    """Verify service cannot exceed pending backlog."""
    cell = GroundCell(cell_id=1, position=np.zeros(3), max_capacity=100.0)
    cell.reset(initial_backlog=10.0)

    # Requested service 25.0 Mb when queue is only 10.0 Mb
    eff_s, dropped = cell.step_queue(served_traffic=25.0, arrival_traffic=0.0)
    assert eff_s == 10.0
    assert cell.queue_backlog == 0.0


def test_queue_capacity_limit():
    """Verify backlog does not exceed max capacity and tracks dropped traffic."""
    cell = GroundCell(cell_id=2, position=np.zeros(3), max_capacity=50.0)
    cell.reset(initial_backlog=40.0)

    # Arrive 30.0 Mb when space is only 10.0 Mb
    eff_s, dropped = cell.step_queue(served_traffic=0.0, arrival_traffic=30.0)
    assert eff_s == 0.0
    assert dropped == 20.0
    assert cell.queue_backlog == 50.0


def test_traffic_model_reset_and_step(test_config):
    """Test full TrafficModel initialization, reset, and step updates."""
    rng = np.random.default_rng(42)
    model = TrafficModel(test_config, rng=rng)

    queues = model.reset()
    assert len(queues) == 20
    assert np.all(queues >= 0.0)

    # Step with zero service
    served_req = np.zeros(20, dtype=np.float32)
    new_q, eff_s, arrivals = model.step(served_req)
    assert len(new_q) == 20
    assert np.all(eff_s == 0.0)
    assert np.all(arrivals >= 0.0)
    assert np.all(new_q >= queues)  # Queues must grow with zero service and positive arrivals

