"""Physical layer transmission, co-channel interference, SINR, and throughput computation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from environment.conflict_resolver import ServiceRequest
from environment.channel_model import ChannelModel


@dataclass
class BeamTransmissionMetrics:
    """Detailed transmission telemetry for an individual beam."""
    sat_id: int
    beam_id: int
    cell_id: int
    target_cell_index: Optional[int]
    is_retained: bool
    is_active: bool
    tx_power_w: float
    rx_power_w: float
    interference_w: float
    noise_power_w: float
    sinr_linear: float
    sinr_db: float
    rate_mbps: float
    potential_data_mb: float
    served_data_mb: float
    rb_start: int
    rb_width: int


@dataclass
class TransmissionResult:
    """Aggregated results across all satellites and cells for a scheduling slot."""
    beam_metrics: Dict[tuple[int, int], BeamTransmissionMetrics] = field(default_factory=dict)
    cell_served_mb: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    total_served_mb: float = 0.0
    total_interference_w: float = 0.0
    max_sinr_db: float = -100.0
    min_sinr_db: float = 100.0
    mean_sinr_db: float = 0.0


class TransmissionModel:
    """
    Physical downlink transmission model.
    Calculates received signal power, cross-satellite co-channel interference
    on overlapping RBs, receiver SINR, achievable Shannon rate, and traffic
    delivery capped strictly at pending queue backlog.
    """

    def __init__(self, config: Dict[str, Any], channel_model: ChannelModel):
        """
        Initialize transmission model.

        Args:
            config: Full configuration dictionary.
            channel_model: ChannelModel instance for antenna gains and noise parameters.
        """
        self.config = config
        self.channel_model = channel_model

        net_cfg = config.get("network", {})
        orb_cfg = config.get("orbit", {})

        self.time_step = float(net_cfg.get("time_step_sec", 1.0))
        self.num_cells = int(net_cfg.get("num_ground_cells", 20))
        self.max_tx_power_w = float(orb_cfg.get("satellite_tx_power_w", 40.0))

    def compute_rb_overlap(self, start_a: int, width_a: int, start_b: int, width_b: int) -> int:
        """Compute the number of overlapping RBs between two chunk allocations."""
        if width_a <= 0 or width_b <= 0:
            return 0
        end_a = start_a + width_a
        end_b = start_b + width_b
        return max(0, min(end_a, end_b) - max(start_a, start_b))

    def execute_transmissions(
        self,
        retained_requests: List[ServiceRequest],
        rejected_requests: List[ServiceRequest],
        satellite_positions_km: Dict[int, np.ndarray],
        cell_positions_km: np.ndarray,
        queue_backlogs_mb: np.ndarray,
    ) -> TransmissionResult:
        """
        Simulate physical downlink transmissions for the scheduling slot.

        Args:
            retained_requests: List of retained ServiceRequests from ConflictResolver.
            rejected_requests: List of rejected ServiceRequests.
            satellite_positions_km: Dict mapping sat_id to [x, y, z] position in km.
            cell_positions_km: Array of shape [num_cells, 3].
            queue_backlogs_mb: Array of shape [num_cells] with pending queue backlog (Mb).

        Returns:
            TransmissionResult containing per-beam and per-cell metrics.
        """
        result = TransmissionResult()
        result.cell_served_mb = np.zeros(self.num_cells, dtype=np.float32)

        # 1. Process rejected beams first (zero physical transmission)
        for req in rejected_requests:
            key = (req.sat_id, req.beam_id)
            result.beam_metrics[key] = BeamTransmissionMetrics(
                sat_id=req.sat_id,
                beam_id=req.beam_id,
                cell_id=req.cell_id,
                target_cell_index=req.target_cell_index,
                is_retained=False,
                is_active=False,
                tx_power_w=0.0,
                rx_power_w=0.0,
                interference_w=0.0,
                noise_power_w=0.0,
                sinr_linear=0.0,
                sinr_db=-100.0,
                rate_mbps=0.0,
                potential_data_mb=0.0,
                served_data_mb=0.0,
                rb_start=req.rb_start,
                rb_width=0,
            )

        if not retained_requests:
            return result

        # 2. Compute co-channel interference for each retained beam
        # Build map of retained transmissions
        active_retained: List[Dict[str, Any]] = []
        for req in retained_requests:
            c_idx = req.target_cell_index
            if c_idx is None:
                continue
            sat_pos = satellite_positions_km[req.sat_id]
            target_cell_pos = cell_positions_km[c_idx]
            tx_power = float(req.power_share * self.max_tx_power_w)

            active_retained.append(
                {
                    "req": req,
                    "sat_id": req.sat_id,
                    "beam_id": req.beam_id,
                    "cell_idx": c_idx,
                    "tx_power": tx_power,
                    "sat_pos": sat_pos,
                    "cell_pos": target_cell_pos,
                    "rb_start": req.rb_start,
                    "rb_width": req.rb_width,
                }
            )

        # Potential delivery per retained beam
        potential_deliveries: Dict[tuple[int, int], float] = {}
        active_metrics: Dict[tuple[int, int], BeamTransmissionMetrics] = {}
        sinr_db_list: List[float] = []

        for item in active_retained:
            req = item["req"]
            key = (req.sat_id, req.beam_id)
            c_idx = item["cell_idx"]
            sat_pos = item["sat_pos"]
            cell_pos = item["cell_pos"]
            tx_pwr = item["tx_power"]
            w_rbs = max(1, item["rb_width"])
            s_rbs = item["rb_start"]

            # Desired received signal power
            h_desired = self.channel_model.compute_channel_gain(sat_pos, cell_pos)
            rx_power = tx_pwr * h_desired

            # Co-channel interference from all OTHER retained beams with overlapping RBs
            total_interf = 0.0
            for other in active_retained:
                # Same beam or same satellite cannot interfere with itself
                if other["sat_id"] == req.sat_id and other["beam_id"] == req.beam_id:
                    continue

                overlap_rbs = self.compute_rb_overlap(
                    s_rbs, w_rbs, other["rb_start"], other["rb_width"]
                )
                if overlap_rbs > 0 and other["tx_power"] > 0.0:
                    # Interfering satellite's channel gain into target cell c_idx,
                    # while its beam is steered toward other['cell_pos']
                    h_interf = self.channel_model.compute_channel_gain(
                        sat_pos_km=other["sat_pos"],
                        cell_pos_km=cell_pos,
                        steered_cell_pos_km=other["cell_pos"],
                    )
                    # Power spectral density fraction
                    interf_share = overlap_rbs / float(w_rbs)
                    p_interf = other["tx_power"] * (overlap_rbs / float(other["rb_width"])) * h_interf
                    total_interf += p_interf

            # Noise power over allocated RBs
            noise_power = self.channel_model.noise_power_per_rb * w_rbs

            # SINR
            sinr_lin = self.channel_model.compute_sinr(
                rx_signal_power_watts=rx_power,
                interference_power_watts=total_interf,
                num_rbs=w_rbs,
            )
            sinr_db = 10.0 * np.log10(max(1e-10, sinr_lin))
            sinr_db_list.append(sinr_db)

            # Shannon achievable rate (Mbps)
            rate_mbps = self.channel_model.compute_achievable_rate_mbps(sinr_lin, num_rbs=w_rbs)
            potential_data = rate_mbps * self.time_step  # Megabits

            potential_deliveries[key] = potential_data
            result.total_interference_w += total_interf

            active_metrics[key] = BeamTransmissionMetrics(
                sat_id=req.sat_id,
                beam_id=req.beam_id,
                cell_id=req.cell_id,
                target_cell_index=c_idx,
                is_retained=True,
                is_active=True,
                tx_power_w=tx_pwr,
                rx_power_w=rx_power,
                interference_w=total_interf,
                noise_power_w=noise_power,
                sinr_linear=sinr_lin,
                sinr_db=sinr_db,
                rate_mbps=rate_mbps,
                potential_data_mb=potential_data,
                served_data_mb=0.0,  # Will be capped below
                rb_start=s_rbs,
                rb_width=w_rbs,
            )

        # 3. Traffic Capping per Cell: Ensure sum(served) <= pending queue Q_c(t)
        # Group active retained beams by cell
        beams_by_cell: Dict[int, List[tuple[int, int]]] = {c: [] for c in range(self.num_cells)}
        for key, m in active_metrics.items():
            if m.target_cell_index is not None:
                beams_by_cell[m.target_cell_index].append(key)

        for c_idx, beam_keys in beams_by_cell.items():
            if not beam_keys:
                continue
            cell_queue = float(queue_backlogs_mb[c_idx])
            tot_pot = sum(potential_deliveries[k] for k in beam_keys)

            if tot_pot <= 0.0 or cell_queue <= 0.0:
                for k in beam_keys:
                    active_metrics[k].served_data_mb = 0.0
                result.cell_served_mb[c_idx] = 0.0
            elif tot_pot <= cell_queue:
                # Fully served
                for k in beam_keys:
                    active_metrics[k].served_data_mb = potential_deliveries[k]
                result.cell_served_mb[c_idx] = tot_pot
            else:
                # Backlog is smaller than transmission capacity: scale proportionally
                scale = cell_queue / tot_pot
                for k in beam_keys:
                    active_metrics[k].served_data_mb = potential_deliveries[k] * scale
                result.cell_served_mb[c_idx] = cell_queue

        # Aggregate total served
        result.total_served_mb = float(np.sum(result.cell_served_mb))

        # Merge active metrics into result
        for k, m in active_metrics.items():
            result.beam_metrics[k] = m

        if sinr_db_list:
            result.max_sinr_db = float(np.max(sinr_db_list))
            result.min_sinr_db = float(np.min(sinr_db_list))
            result.mean_sinr_db = float(np.mean(sinr_db_list))

        return result

