"""Unit tests for resource allocation constraints: beams, RBs, power conservation."""

import pytest
import numpy as np
import yaml
from environment.resource_manager import ResourceManager, BeamAction


@pytest.fixture
def test_config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_invisible_cell_rejection(test_config):
    """Test that selecting an invisible cell forces the beam to inactive."""
    rm = ResourceManager(test_config)
    visible_cells = [3, 7]  # 0-based cell indices

    # Beam 0 selects visible cell index 3 (cell_id = 4)
    # Beam 1 selects invisible cell index 9 (cell_id = 10)
    raw_actions = [
        {"cell_id": 4, "rb_start": 0, "rb_width": 2, "power_share": 0.5},
        {"cell_id": 10, "rb_start": 2, "rb_width": 2, "power_share": 0.5},
    ]

    alloc = rm.validate_action(sat_id=0, raw_beam_actions=raw_actions, visible_cell_indices=visible_cells)

    assert alloc.beam_allocations[0].is_active is True
    assert alloc.beam_allocations[0].target_cell_index == 3
    # Invisible beam forced inactive
    assert alloc.beam_allocations[1].is_active is False
    assert alloc.beam_allocations[1].target_cell_index is None
    assert alloc.beam_allocations[1].power_share == 0.0
    assert alloc.beam_allocations[1].rb_width == 0


def test_intra_satellite_rb_exclusivity(test_config):
    """Test that two active beams on the same satellite cannot share overlapping RBs."""
    rm = ResourceManager(test_config)
    visible_cells = [1, 2]

    # Both beams request overlapping RBs: [0, 1, 2, 3] and [2, 3, 4]
    raw_actions = [
        {"cell_id": 2, "rb_start": 0, "rb_width": 4, "power_share": 0.4},
        {"cell_id": 3, "rb_start": 2, "rb_width": 3, "power_share": 0.4},
    ]

    alloc = rm.validate_action(sat_id=0, raw_beam_actions=raw_actions, visible_cell_indices=visible_cells)

    b0 = alloc.beam_allocations[0]
    b1 = alloc.beam_allocations[1]

    assert b0.is_active is True
    assert b1.is_active is True

    # Must have strictly disjoint RB allocations
    rbs_0 = set(range(b0.rb_start, b0.rb_start + b0.rb_width))
    rbs_1 = set(range(b1.rb_start, b1.rb_start + b1.rb_width))
    assert len(rbs_0.intersection(rbs_1)) == 0


def test_power_conservation(test_config):
    """Test sum(p_{i,k}) + p_{i,0} == 1.0 constraint with over-allocation and scaling."""
    rm = ResourceManager(test_config)
    visible_cells = [0, 1]

    # Beams request 0.8 and 0.6 -> sum = 1.4 > 1.0
    raw_actions = [
        {"cell_id": 1, "rb_start": 0, "rb_width": 2, "power_share": 0.8},
        {"cell_id": 2, "rb_start": 2, "rb_width": 2, "power_share": 0.6},
    ]

    alloc = rm.validate_action(sat_id=0, raw_beam_actions=raw_actions, visible_cell_indices=visible_cells)

    p0 = alloc.beam_allocations[0].power_share
    p1 = alloc.beam_allocations[1].power_share
    p_reserved = alloc.reserved_power_share

    # Scaled down to 0.8/1.4 and 0.6/1.4
    assert pytest.approx(p0 + p1 + p_reserved, 1e-6) == 1.0
    assert pytest.approx(p0, 1e-6) == 0.8 / 1.4
    assert pytest.approx(p1, 1e-6) == 0.6 / 1.4
    assert p_reserved == 0.0


def test_rb_overlap_calculation(test_config):
    """Verify compute_rb_overlap calculation."""
    rm = ResourceManager(test_config)

    # Chunk [0..3] (width 4) vs [2..4] (width 3) -> overlap [2, 3] = 2 RBs
    assert rm.compute_rb_overlap(start_a=0, width_a=4, start_b=2, width_b=3) == 2

    # Chunk [0..1] (width 2) vs [4..7] (width 4) -> disjoint = 0 RBs
    assert rm.compute_rb_overlap(start_a=0, width_a=2, start_b=4, width_b=4) == 0

    # Inactive beam width 0 -> 0 RBs
    assert rm.compute_rb_overlap(start_a=0, width_a=0, start_b=0, width_b=4) == 0

