"""Unit tests for downlink physical transmission, interference, and SINR."""

import pytest
import numpy as np
import yaml
from environment.channel_model import ChannelModel
from environment.transmission_model import TransmissionModel
from environment.conflict_resolver import ServiceRequest


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_zero_interference_with_disjoint_rbs(test_config):
    """When two satellites use disjoint RBs, interference must be exactly zero."""
    channel = ChannelModel(test_config)
    tx_model = TransmissionModel(test_config, channel)

    sat_positions = {
        0: np.array([0.0, 0.0, 6921.0]),  # 550 km altitude above pole
        1: np.array([50.0, 0.0, 6921.0]),
    }
    cell_positions = np.array([[0.0, 0.0, 6371.0], [50.0, 0.0, 6371.0]])
    queue_backlogs = np.array([50.0, 50.0], dtype=np.float32)

    # Sat 0 serves Cell 0 on RBs [0..3], Sat 1 serves Cell 1 on RBs [4..7]
    req0 = ServiceRequest(
        sat_id=0, shell_id=0, beam_id=0, cell_id=1, power_share=0.5,
        rb_start=0, rb_width=4, channel_gain=1e-11, propagation_delay_sec=0.002
    )
    req1 = ServiceRequest(
        sat_id=1, shell_id=0, beam_id=0, cell_id=2, power_share=0.5,
        rb_start=4, rb_width=4, channel_gain=1e-11, propagation_delay_sec=0.002
    )

    res = tx_model.execute_transmissions(
        retained_requests=[req0, req1],
        rejected_requests=[],
        satellite_positions_km=sat_positions,
        cell_positions_km=cell_positions,
        queue_backlogs_mb=queue_backlogs,
    )

    m0 = res.beam_metrics[(0, 0)]
    m1 = res.beam_metrics[(1, 0)]

    assert m0.interference_w == 0.0
    assert m1.interference_w == 0.0
    assert res.total_interference_w == 0.0
    assert m0.served_data_mb > 0.0
    assert m1.served_data_mb > 0.0


def test_positive_interference_with_overlapping_rbs(test_config):
    """When two satellites use overlapping RBs, interference must be strictly positive."""
    channel = ChannelModel(test_config)
    tx_model = TransmissionModel(test_config, channel)

    sat_positions = {
        0: np.array([0.0, 0.0, 6921.0]),
        1: np.array([100.0, 0.0, 6921.0]),
    }
    cell_positions = np.array([[0.0, 0.0, 6371.0], [100.0, 0.0, 6371.0]])
    queue_backlogs = np.array([50.0, 50.0], dtype=np.float32)

    # Both satellites allocate overlapping RBs [0..3]
    req0 = ServiceRequest(
        sat_id=0, shell_id=0, beam_id=0, cell_id=1, power_share=0.5,
        rb_start=0, rb_width=4, channel_gain=1e-11, propagation_delay_sec=0.002
    )
    req1 = ServiceRequest(
        sat_id=1, shell_id=0, beam_id=0, cell_id=2, power_share=0.5,
        rb_start=0, rb_width=4, channel_gain=1e-11, propagation_delay_sec=0.002
    )

    res = tx_model.execute_transmissions(
        retained_requests=[req0, req1],
        rejected_requests=[],
        satellite_positions_km=sat_positions,
        cell_positions_km=cell_positions,
        queue_backlogs_mb=queue_backlogs,
    )

    m0 = res.beam_metrics[(0, 0)]
    assert m0.interference_w > 0.0
    assert res.total_interference_w > 0.0


def test_service_capped_strictly_at_backlog(test_config):
    """Total served traffic for a cell must never exceed its backlog."""
    channel = ChannelModel(test_config)
    tx_model = TransmissionModel(test_config, channel)

    sat_positions = {0: np.array([0.0, 0.0, 6921.0])}
    cell_positions = np.array([[0.0, 0.0, 6371.0]])
    # Very small queue backlog: 1.5 Mb
    queue_backlogs = np.array([1.5], dtype=np.float32)

    req0 = ServiceRequest(
        sat_id=0, shell_id=0, beam_id=0, cell_id=1, power_share=0.8,
        rb_start=0, rb_width=8, channel_gain=1e-11, propagation_delay_sec=0.002
    )

    res = tx_model.execute_transmissions(
        retained_requests=[req0],
        rejected_requests=[],
        satellite_positions_km=sat_positions,
        cell_positions_km=cell_positions,
        queue_backlogs_mb=queue_backlogs,
    )

    m0 = res.beam_metrics[(0, 0)]
    # Potential transmission capability is high (> 100 Mb), but served must be capped at 1.5 Mb
    assert m0.potential_data_mb > 1.5
    assert pytest.approx(m0.served_data_mb, 1e-5) == 1.5
    assert pytest.approx(res.cell_served_mb[0], 1e-5) == 1.5


def test_rejected_beams_deliver_zero(test_config):
    """Rejected beams must deliver zero traffic and contribute zero interference."""
    channel = ChannelModel(test_config)
    tx_model = TransmissionModel(test_config, channel)

    sat_positions = {0: np.array([0.0, 0.0, 6921.0])}
    cell_positions = np.array([[0.0, 0.0, 6371.0]])
    queue_backlogs = np.array([50.0], dtype=np.float32)

    rejected_req = ServiceRequest(
        sat_id=0, shell_id=0, beam_id=0, cell_id=1, power_share=0.5,
        rb_start=0, rb_width=4, channel_gain=1e-11, propagation_delay_sec=0.002
    )

    res = tx_model.execute_transmissions(
        retained_requests=[],
        rejected_requests=[rejected_req],
        satellite_positions_km=sat_positions,
        cell_positions_km=cell_positions,
        queue_backlogs_mb=queue_backlogs,
    )

    m = res.beam_metrics[(0, 0)]
    assert m.is_retained is False
    assert m.is_active is False
    assert m.tx_power_w == 0.0
    assert m.served_data_mb == 0.0
    assert res.total_served_mb == 0.0

