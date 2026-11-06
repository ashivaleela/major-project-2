"""Traffic and queue model for ground cells in the multi-shell LEO network."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class GroundCell:
    """Represents a terrestrial ground cell with dynamic traffic demand."""
    cell_id: int
    position: np.ndarray  # [x, y, z] in km in ECEF Cartesian coordinates
    is_hotspot: bool = False
    mean_arrival_rate: float = 3.5  # Mbps (Mbit/s)
    queue_backlog: float = 0.0  # Current pending traffic in Megabits (Mb)
    max_capacity: float = 200.0  # Max queue buffer capacity (Mb)
    cumulative_arrived: float = 0.0
    cumulative_served: float = 0.0
    cumulative_dropped: float = 0.0
    history_served: List[float] = field(default_factory=list)
    history_backlog: List[float] = field(default_factory=list)

    def reset(self, initial_backlog: float = 0.0) -> None:
        """Reset the cell traffic queue and statistics."""
        self.queue_backlog = float(np.clip(initial_backlog, 0.0, self.max_capacity))
        self.cumulative_arrived = self.queue_backlog
        self.cumulative_served = 0.0
        self.cumulative_dropped = 0.0
        self.history_served.clear()
        self.history_backlog.clear()
        self.history_backlog.append(self.queue_backlog)

    def step_queue(self, served_traffic: float, arrival_traffic: float) -> tuple[float, float]:
        """
        Execute deterministic queue update:
            Q_c(t+1) = max(0, Q_c(t) - x_c(t)) + A_c(t)
        
        Args:
            served_traffic: Amount of successfully transmitted traffic (Mb).
            arrival_traffic: Amount of newly arrived traffic (Mb).

        Returns:
            Tuple of (effective_served_mb, dropped_traffic_mb).
        """
        effective_served = float(min(self.queue_backlog, max(0.0, served_traffic)))
        post_service_backlog = self.queue_backlog - effective_served

        new_backlog = post_service_backlog + max(0.0, arrival_traffic)
        dropped = 0.0
        if new_backlog > self.max_capacity:
            dropped = new_backlog - self.max_capacity
            new_backlog = self.max_capacity

        self.queue_backlog = float(new_backlog)
        self.cumulative_served += effective_served
        self.cumulative_arrived += arrival_traffic
        self.cumulative_dropped += dropped
        self.history_served.append(effective_served)
        self.history_backlog.append(self.queue_backlog)

        return effective_served, dropped


class TrafficModel:
    """Manages ground cells and their traffic generation processes."""

    R_EARTH_KM = 6371.0

    def __init__(self, config: Dict[str, Any], rng: Optional[np.random.Generator] = None):
        """
        Initialize ground cells with configured distribution.

        Args:
            config: Full configuration dictionary containing 'network' and 'traffic' sections.
            rng: NumPy random generator for reproducibility.
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.get("experiment", {}).get("seed", 42))

        net_cfg = config.get("network", {})
        traf_cfg = config.get("traffic", {})

        self.num_cells = int(net_cfg.get("num_ground_cells", 20))
        self.time_step = float(net_cfg.get("time_step_sec", 1.0))
        self.hotspot_ratio = float(traf_cfg.get("hotspot_cell_ratio", 0.25))
        self.hotspot_arrival_rate = float(traf_cfg.get("hotspot_arrival_rate_mbps", 12.0))
        self.regular_arrival_rate = float(traf_cfg.get("regular_arrival_rate_mbps", 3.5))
        self.burst_prob = float(traf_cfg.get("burst_probability", 0.15))
        self.burst_mult = float(traf_cfg.get("burst_multiplier", 2.5))
        self.initial_queue_max = float(traf_cfg.get("initial_queue_max_mb", 30.0))
        self.max_capacity = float(traf_cfg.get("max_queue_capacity_mb", 200.0))
        self.arrival_dist = str(traf_cfg.get("arrival_distribution", "poisson"))

        self.cells: List[GroundCell] = []
        self._init_cells()

    def _init_cells(self) -> None:
        """Create ground cells distributed across an operational mid-latitude ground region."""
        self.cells.clear()

        # Operational coverage center: Latitude 40.0 deg N, Longitude aligned with sub-satellite pass
        phi_0 = np.radians(40.0)
        inc_ref = np.radians(53.0)
        u_ref = np.arcsin(np.sin(phi_0) / np.sin(inc_ref))
        lam_0 = np.arctan2(np.cos(inc_ref) * np.sin(u_ref), np.cos(u_ref))

        # Local East, North, Up unit vectors on Earth sphere
        u_vec = np.array([np.cos(phi_0) * np.cos(lam_0), np.cos(phi_0) * np.sin(lam_0), np.sin(phi_0)])
        e_vec = np.array([-np.sin(lam_0), np.cos(lam_0), 0.0])
        n_vec = np.array([-np.sin(phi_0) * np.cos(lam_0), -np.sin(phi_0) * np.sin(lam_0), np.cos(phi_0)])
        r_center = self.R_EARTH_KM * u_vec

        grid_dim = int(np.ceil(np.sqrt(self.num_cells)))
        spacing_km = 90.0  # 90 km spacing between ground cell centers

        num_hotspots = max(1, int(round(self.num_cells * self.hotspot_ratio)))
        hotspot_indices = set(self.rng.choice(self.num_cells, size=num_hotspots, replace=False))

        for c_id in range(self.num_cells):
            row = c_id // grid_dim
            col = c_id % grid_dim
            dx = (col - grid_dim / 2.0 + 0.5) * spacing_km
            dy = (row - grid_dim / 2.0 + 0.5) * spacing_km

            # Project onto sphere
            pt = r_center + dx * e_vec + dy * n_vec
            pt_norm = pt / np.linalg.norm(pt)
            pos = self.R_EARTH_KM * pt_norm

            is_hotspot = c_id in hotspot_indices
            mean_rate = self.hotspot_arrival_rate if is_hotspot else self.regular_arrival_rate

            cell = GroundCell(
                cell_id=c_id,
                position=pos,
                is_hotspot=is_hotspot,
                mean_arrival_rate=mean_rate,
                max_capacity=self.max_capacity,
            )
            self.cells.append(cell)

    def reset(self, initial_queues: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset all cells' queues.

        Args:
            initial_queues: Optional array of initial backlog values (Mb).

        Returns:
            Array of shape [num_cells] with initial backlogs.
        """
        backlogs = []
        for i, cell in enumerate(self.cells):
            if initial_queues is not None:
                init_val = float(initial_queues[i])
            else:
                init_val = float(self.rng.uniform(5.0, self.initial_queue_max))
            cell.reset(initial_backlog=init_val)
            backlogs.append(cell.queue_backlog)
        return np.array(backlogs, dtype=np.float32)

    def generate_arrivals(self) -> np.ndarray:
        """
        Generate random traffic arrivals for each cell in the current slot.
        Arrival traffic = rate (Mbps) * time_step (s) -> Megabits.

        Returns:
            NumPy array of shape [num_cells] with arrival amounts in Mb.
        """
        arrivals = np.zeros(self.num_cells, dtype=np.float32)

        for i, cell in enumerate(self.cells):
            mean_arrival_mb = cell.mean_arrival_rate * self.time_step

            # Check for burst
            if self.rng.random() < self.burst_prob:
                mean_arrival_mb *= self.burst_mult

            if self.arrival_dist == "poisson":
                scale = 10.0
                lam = max(0.1, mean_arrival_mb * scale)
                arrivals[i] = float(self.rng.poisson(lam) / scale)
            else:
                arrivals[i] = float(max(0.0, self.rng.normal(mean_arrival_mb, mean_arrival_mb * 0.3)))

        return arrivals

    def step(self, served_traffic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update queues given served traffic across all cells.

        Args:
            served_traffic: Array of shape [num_cells] containing delivered data (Mb).

        Returns:
            Tuple of (new_queues, effective_served, arrivals).
        """
        arrivals = self.generate_arrivals()
        effective_served = np.zeros(self.num_cells, dtype=np.float32)
        new_queues = np.zeros(self.num_cells, dtype=np.float32)

        for i, cell in enumerate(self.cells):
            srv = float(served_traffic[i]) if i < len(served_traffic) else 0.0
            arr = float(arrivals[i])
            eff_s, _ = cell.step_queue(served_traffic=srv, arrival_traffic=arr)
            effective_served[i] = eff_s
            new_queues[i] = cell.queue_backlog

        return new_queues, effective_served, arrivals

    def get_queues(self) -> np.ndarray:
        """Return current queue backlogs of all cells (Mb)."""
        return np.array([cell.queue_backlog for cell in self.cells], dtype=np.float32)

    def get_positions(self) -> np.ndarray:
        """Return positions of all cells (km), shape [num_cells, 3]."""
        return np.array([cell.position for cell in self.cells], dtype=np.float64)

    def get_hotspot_flags(self) -> np.ndarray:
        """Return boolean hotspot flags for all cells."""
        return np.array([cell.is_hotspot for cell in self.cells], dtype=bool)
