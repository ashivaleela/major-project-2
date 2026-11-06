"""Satellite network representation and multi-shell orbital mechanics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from environment.channel_model import ChannelModel


@dataclass
class Satellite:
    """Represents an individual LEO satellite belonging to an orbital shell."""
    sat_id: int
    shell_id: int
    altitude_km: float
    inclination_deg: float
    raan_deg: float
    true_anomaly_deg: float
    max_tx_power_w: float = 40.0
    num_beams: int = 2
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    previous_beam_cell: List[int] = field(default_factory=list)
    visible_cells: List[int] = field(default_factory=list)
    is_operational: bool = True

    channel_gains: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def reset_state(self, initial_anomaly: float) -> None:
        """Reset satellite state and trajectory to initial anomaly."""
        self.true_anomaly_deg = float(initial_anomaly % 360.0)
        self.previous_beam_cell = [0] * self.num_beams
        self.visible_cells.clear()
        self.is_operational = True


class SatelliteNetwork:
    """Manages multi-shell LEO satellite constellation dynamics and visibility."""

    MU_EARTH = 398600.4418  # Earth gravitational parameter km^3/s^2
    R_EARTH_KM = 6371.0
    OMEGA_EARTH = 7.292115e-5  # Earth rotation rate rad/s

    def __init__(
        self,
        config: Dict[str, Any],
        channel_model: ChannelModel,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize multi-shell constellation.

        Args:
            config: Full configuration dictionary.
            channel_model: ChannelModel instance for visibility and path-loss calculations.
            rng: NumPy random generator.
        """
        self.config = config
        self.channel_model = channel_model
        self.rng = rng if rng is not None else np.random.default_rng(config.get("experiment", {}).get("seed", 42))

        net_cfg = config.get("network", {})
        orb_cfg = config.get("orbit", {})

        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.num_shells = int(net_cfg.get("num_shells", 2))
        self.sats_per_shell = list(net_cfg.get("satellites_per_shell", [4, 4]))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_ground_cells = int(net_cfg.get("num_ground_cells", 20))
        self.time_step = float(net_cfg.get("time_step_sec", 1.0))

        self.shell_altitudes = list(orb_cfg.get("shell_altitudes_km", [550.0, 1100.0]))
        self.shell_inclinations = list(orb_cfg.get("inclination_deg", [53.0, 70.0]))
        self.max_tx_power_w = float(orb_cfg.get("satellite_tx_power_w", 40.0))

        self.satellites: List[Satellite] = []
        self._init_constellation()

        self.current_step = 0
        self.earth_rotation_angle = 0.0

    def _init_constellation(self) -> None:
        """Create satellites partitioned across shells with phased orbital planes over target region."""
        self.satellites.clear()
        sat_id = 0

        # Phasing reference for latitude 40 N
        phi_0 = np.radians(40.0)

        for shell_idx in range(self.num_shells):
            count = self.sats_per_shell[shell_idx]
            alt = self.shell_altitudes[shell_idx]
            inc = self.shell_inclinations[shell_idx]

            inc_rad = np.radians(inc)
            # Reference anomaly when passing latitude 40 deg
            u_base = np.degrees(np.arcsin(np.clip(np.sin(phi_0) / np.sin(inc_rad), -1.0, 1.0)))

            for i in range(count):
                # Stagger anomalies across ground coverage zone
                anomaly_offset = (i - (count - 1) / 2.0) * 6.5
                initial_anomaly = (u_base + anomaly_offset) % 360.0
                raan = 0.0  # Aligned ground track plane

                sat = Satellite(
                    sat_id=sat_id,
                    shell_id=shell_idx,
                    altitude_km=alt,
                    inclination_deg=inc,
                    raan_deg=raan,
                    true_anomaly_deg=initial_anomaly,
                    max_tx_power_w=self.max_tx_power_w,
                    num_beams=self.beams_per_satellite,
                    previous_beam_cell=[0] * self.beams_per_satellite,
                )
                self.satellites.append(sat)
                sat_id += 1

    def compute_keplerian_position(
        self,
        altitude_km: float,
        inclination_deg: float,
        raan_deg: float,
        true_anomaly_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Cartesian ECEF position and velocity of a circular LEO orbit.
        """
        r = self.R_EARTH_KM + altitude_km
        v_mag = np.sqrt(self.MU_EARTH / r)

        u_rad = np.radians(true_anomaly_deg)
        inc_rad = np.radians(inclination_deg)
        raan_rad = np.radians(raan_deg - np.degrees(self.earth_rotation_angle))

        # Position in orbital plane
        x_orb = r * np.cos(u_rad)
        y_orb = r * np.sin(u_rad)
        vx_orb = -v_mag * np.sin(u_rad)
        vy_orb = v_mag * np.cos(u_rad)

        # Rotation matrix to ECEF: R_z(raan) * R_x(inc)
        cos_o, sin_o = np.cos(raan_rad), np.sin(raan_rad)
        cos_i, sin_i = np.cos(inc_rad), np.sin(inc_rad)

        x = cos_o * x_orb - sin_o * cos_i * y_orb
        y = sin_o * x_orb + cos_o * cos_i * y_orb
        z = sin_i * y_orb

        vx = cos_o * vx_orb - sin_o * cos_i * vy_orb
        vy = sin_o * vx_orb + cos_o * cos_i * vy_orb
        vz = sin_i * vy_orb

        return np.array([x, y, z], dtype=np.float64), np.array([vx, vy, vz], dtype=np.float64)

    def reset(self) -> None:
        """Reset constellation states and orbital anomalies for a new episode."""
        self.current_step = 0
        self.earth_rotation_angle = 0.0

        phi_0 = np.radians(40.0)

        for sat in self.satellites:
            inc_rad = np.radians(sat.inclination_deg)
            u_base = np.degrees(np.arcsin(np.clip(np.sin(phi_0) / np.sin(inc_rad), -1.0, 1.0)))
            count = self.sats_per_shell[sat.shell_id]
            sub_idx = sat.sat_id if sat.shell_id == 0 else (sat.sat_id - self.sats_per_shell[0])
            anomaly_offset = (sub_idx - (count - 1) / 2.0) * 6.5
            base_anomaly = (u_base + anomaly_offset) % 360.0

            sat.reset_state(initial_anomaly=base_anomaly)
            pos, vel = self.compute_keplerian_position(
                altitude_km=sat.altitude_km,
                inclination_deg=sat.inclination_deg,
                raan_deg=sat.raan_deg,
                true_anomaly_deg=sat.true_anomaly_deg,
            )
            sat.position = pos
            sat.velocity = vel

    def step_orbits(self) -> None:
        """Advance satellite positions along their orbits by time_step seconds."""
        self.current_step += 1
        self.earth_rotation_angle += self.OMEGA_EARTH * self.time_step

        for sat in self.satellites:
            r = self.R_EARTH_KM + sat.altitude_km
            n_rad_s = np.sqrt(self.MU_EARTH / (r**3))
            delta_anomaly_deg = np.degrees(n_rad_s * self.time_step)

            sat.true_anomaly_deg = (sat.true_anomaly_deg + delta_anomaly_deg) % 360.0
            pos, vel = self.compute_keplerian_position(
                altitude_km=sat.altitude_km,
                inclination_deg=sat.inclination_deg,
                raan_deg=sat.raan_deg,
                true_anomaly_deg=sat.true_anomaly_deg,
            )
            sat.position = pos
            sat.velocity = vel

    def update_visibility_and_channels(self, cell_positions_km: np.ndarray) -> np.ndarray:
        """
        Compute visibility and channel gains between all satellites and ground cells.

        Returns:
            visibility_matrix: Boolean array of shape [num_satellites, num_cells].
        """
        num_cells = len(cell_positions_km)
        visibility = np.zeros((self.num_satellites, num_cells), dtype=bool)

        for sat in self.satellites:
            sat.visible_cells.clear()
            sat.channel_gains = np.zeros(num_cells, dtype=np.float64)

            if not getattr(sat, 'is_operational', True):
                continue

            for c_id in range(num_cells):
                cell_pos = cell_positions_km[c_id]
                elev = self.channel_model.compute_elevation_angle(sat.position, cell_pos)
                if elev >= self.channel_model.min_elevation_deg:
                    visibility[sat.sat_id, c_id] = True
                    sat.visible_cells.append(c_id)
                    h = self.channel_model.compute_channel_gain(sat.position, cell_pos)
                    sat.channel_gains[c_id] = h
                else:
                    sat.channel_gains[c_id] = 1e-25

        return visibility

    # Satellite failure and recovery

    def set_satellite_operational(self, sat_id: int, operational: bool) -> None:
        """Change the operational status of a satellite."""
        satellite = next(
            (sat for sat in self.satellites if sat.sat_id == sat_id),
            None
        )
        if satellite is None:
            raise ValueError(f"Invalid satellite ID: {sat_id}")

        satellite.is_operational = bool(operational)

        if not satellite.is_operational:
            satellite.visible_cells.clear()
            satellite.channel_gains = np.zeros_like(
                satellite.channel_gains
            )
            satellite.previous_beam_cell = [0] * satellite.num_beams

    def fail_satellite(self, sat_id: int) -> None:
        """Simulate satellite failure."""
        self.set_satellite_operational(sat_id, False)

    def restore_satellite(self, sat_id: int) -> None:
        """Restore a failed satellite."""
        self.set_satellite_operational(sat_id, True)

    def get_shell_satellites(self, shell_id: int) -> List[Satellite]:
        """Get all satellites belonging to a specific orbital shell."""
        return [sat for sat in self.satellites if sat.shell_id == shell_id]
