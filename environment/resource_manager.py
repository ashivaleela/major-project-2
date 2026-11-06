"""Resource validation and constraint enforcement for satellite beams, RBs, and power."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class BeamAction:
    """Action parameters for an individual satellite beam."""
    beam_id: int
    cell_id: int  # 0: Inactive, 1..N: Target ground cell index (cell_id - 1)
    rb_start: int  # Starting RB index in [0, num_rbs - 1]
    rb_width: int  # Chunk width in [1, num_rbs - rb_start], 0 if inactive
    power_share: float  # Normalized power share p_{i,k} in [0.0, 1.0]
    is_active: bool = False

    @property
    def target_cell_index(self) -> Optional[int]:
        """Convert action cell_id to 0-based cell index, or None if inactive."""
        return (self.cell_id - 1) if self.cell_id > 0 else None

    @property
    def occupied_rbs(self) -> set[int]:
        """Set of discrete RB indices occupied by this beam."""
        if not self.is_active or self.rb_width <= 0:
            return set()
        return set(range(self.rb_start, self.rb_start + self.rb_width))


@dataclass
class SatelliteAllocation:
    """Sanitized and physically valid allocation for a single satellite."""
    sat_id: int
    beam_allocations: List[BeamAction]
    reserved_power_share: float  # p_{i,0} in [0, 1]
    is_valid: bool = True
    validation_notes: List[str] = None


class ResourceManager:
    """
    Validates, sanitizes, and enforces intra-satellite resource exclusivity,
    beam feasibility masks, and physical power conservation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize resource constraints from config.

        Args:
            config: Full configuration dictionary.
        """
        self.config = config
        net_cfg = config.get("network", {})
        orb_cfg = config.get("orbit", {})

        self.num_rbs = int(net_cfg.get("num_resource_blocks", 8))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_ground_cells = int(net_cfg.get("num_ground_cells", 20))
        self.max_tx_power_w = float(orb_cfg.get("satellite_tx_power_w", 40.0))

    def compute_rb_overlap(self, start_a: int, width_a: int, start_b: int, width_b: int) -> int:
        """
        Calculate number of overlapping resource blocks between two allocations.

        Returns:
            Number of common RBs (>= 0).
        """
        if width_a <= 0 or width_b <= 0:
            return 0
        end_a = start_a + width_a
        end_b = start_b + width_b
        overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
        return overlap

    def validate_action(
        self,
        sat_id: int,
        raw_beam_actions: List[Dict[str, Any]],
        visible_cell_indices: List[int],
    ) -> SatelliteAllocation:
        """
        Sanitize and enforce physical constraints on proposed beam actions.

        Rules enforced:
        1. Infeasible / invisible cells cannot be selected (forced to inactive).
        2. Inactive beams have rb_width = 0, power_share = 0.
        3. Valid RB boundaries: 0 <= rb_start < num_rbs, 1 <= rb_width <= num_rbs - rb_start.
        4. Intra-satellite RB exclusivity: No two active beams on the same satellite share RBs.
        5. Normalized power conservation: sum(p_{i,k}) + p_{i,0} = 1.0, with p >= 0.

        Args:
            sat_id: ID of the satellite.
            raw_beam_actions: List of dicts with keys: 'cell_id', 'rb_start', 'rb_width', 'power_share'.
            visible_cell_indices: List of 0-based cell indices currently visible to this satellite.

        Returns:
            A sanitized SatelliteAllocation.
        """
        notes: List[str] = []
        sanitized_beams: List[BeamAction] = []
        occupied_satellite_rbs: set[int] = set()

        # Phase 1: Beam target and RB validation
        for b_idx in range(self.beams_per_satellite):
            if b_idx < len(raw_beam_actions):
                raw = raw_beam_actions[b_idx]
                cell_choice = int(raw.get("cell_id", 0))
                rb_start = int(raw.get("rb_start", 0))
                rb_width = int(raw.get("rb_width", 1))
                p_share = float(raw.get("power_share", 0.0))
            else:
                cell_choice = 0
                rb_start = 0
                rb_width = 0
                p_share = 0.0

            # Cell selection validation: 0 is inactive, 1..N correspond to cell 0..N-1
            is_active = False
            if cell_choice > 0:
                cell_idx = cell_choice - 1
                if cell_idx in visible_cell_indices and 0 <= cell_idx < self.num_ground_cells:
                    is_active = True
                else:
                    notes.append(f"Beam {b_idx} selected invisible cell {cell_idx}; forced inactive.")
                    cell_choice = 0
                    is_active = False

            if not is_active:
                sanitized_beams.append(
                    BeamAction(
                        beam_id=b_idx,
                        cell_id=0,
                        rb_start=0,
                        rb_width=0,
                        power_share=0.0,
                        is_active=False,
                    )
                )
                continue

            # RB bounds validation
            rb_start = int(np.clip(rb_start, 0, self.num_rbs - 1))
            max_possible_width = self.num_rbs - rb_start
            rb_width = int(np.clip(rb_width, 1, max_possible_width))

            # Intra-satellite RB exclusivity enforcement
            desired_rbs = set(range(rb_start, rb_start + rb_width))
            conflicting_rbs = desired_rbs.intersection(occupied_satellite_rbs)

            if conflicting_rbs:
                # Find the largest non-conflicting contiguous sub-chunk, or pick from free RBs
                free_rbs = sorted(list(set(range(self.num_rbs)) - occupied_satellite_rbs))
                if free_rbs:
                    # Allocate first available contiguous block from free RBs
                    rb_start = free_rbs[0]
                    # Find how many contiguous free RBs follow
                    curr_width = 1
                    while (rb_start + curr_width) in free_rbs and curr_width < rb_width:
                        curr_width += 1
                    rb_width = curr_width
                    notes.append(
                        f"Beam {b_idx} RB conflict resolved: reassigned to start={rb_start}, width={rb_width}."
                    )
                else:
                    # No free RBs available on this satellite -> force inactive
                    notes.append(f"Beam {b_idx} deactivated due to lack of free RBs.")
                    sanitized_beams.append(
                        BeamAction(
                            beam_id=b_idx,
                            cell_id=0,
                            rb_start=0,
                            rb_width=0,
                            power_share=0.0,
                            is_active=False,
                        )
                    )
                    continue

            # Register occupied RBs
            allocated_rbs = set(range(rb_start, rb_start + rb_width))
            occupied_satellite_rbs.update(allocated_rbs)

            sanitized_beams.append(
                BeamAction(
                    beam_id=b_idx,
                    cell_id=cell_choice,
                    rb_start=rb_start,
                    rb_width=rb_width,
                    power_share=max(0.0, p_share),
                    is_active=True,
                )
            )

        # Phase 2: Power normalization across active beams
        active_indices = [idx for idx, b in enumerate(sanitized_beams) if b.is_active]
        if not active_indices:
            # All beams inactive: 100% reserved power
            for b in sanitized_beams:
                b.power_share = 0.0
            reserved_power = 1.0
        else:
            raw_powers = [sanitized_beams[idx].power_share for idx in active_indices]
            sum_raw = sum(raw_powers)

            if sum_raw > 1.0:
                # Scale down proportionally to fit power budget
                for idx in active_indices:
                    sanitized_beams[idx].power_share = float(sanitized_beams[idx].power_share / sum_raw)
                reserved_power = 0.0
            elif sum_raw <= 0.0:
                # If active beams requested 0 or negative power, divide available power equally
                equal_share = 0.8 / len(active_indices)
                for idx in active_indices:
                    sanitized_beams[idx].power_share = float(equal_share)
                reserved_power = 0.2
            else:
                # Valid sum <= 1.0
                reserved_power = float(max(0.0, 1.0 - sum_raw))

        # Inactive beams strictly have 0 power
        for b in sanitized_beams:
            if not b.is_active:
                b.power_share = 0.0

        return SatelliteAllocation(
            sat_id=sat_id,
            beam_allocations=sanitized_beams,
            reserved_power_share=reserved_power,
            is_valid=True,
            validation_notes=notes,
        )

