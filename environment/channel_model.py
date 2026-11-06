"""Physical wireless channel and link budget model for multi-shell LEO downlinks."""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np


class ChannelModel:
    """
    Computes satellite-to-ground link geometry, path loss, antenna gain patterns,
    atmospheric absorption, propagation delays, and thermal noise.
    """

    C_LIGHT = 299792458.0  # Speed of light in m/s
    BOLTZMANN_K = 1.380649e-23  # Boltzmann constant J/K
    R_EARTH_KM = 6371.0

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize channel parameters from config.

        Args:
            config: Full configuration dictionary.
        """
        self.config = config
        net_cfg = config.get("network", {})
        orb_cfg = config.get("orbit", {})

        self.carrier_freq_ghz = float(net_cfg.get("carrier_frequency_ghz", 20.0))
        self.carrier_freq_hz = self.carrier_freq_ghz * 1e9
        self.rb_bandwidth_mhz = float(net_cfg.get("rb_bandwidth_mhz", 25.0))
        self.rb_bandwidth_hz = self.rb_bandwidth_mhz * 1e6

        self.min_elevation_deg = float(orb_cfg.get("min_elevation_deg", 25.0))
        self.tx_antenna_gain_dbi = float(orb_cfg.get("satellite_antenna_gain_dbi", 32.0))
        self.rx_antenna_gain_dbi = float(orb_cfg.get("ground_antenna_gain_dbi", 38.0))
        self.noise_temp_k = float(orb_cfg.get("noise_temperature_k", 350.0))
        self.efficiency = float(orb_cfg.get("implementation_efficiency", 0.85))

        # Beam pattern parameters
        self.beam_3db_width_deg = 3.5  # 3-dB beamwidth
        self.min_sidelobe_gain_dbi = -10.0  # Minimum floor for antenna side-lobes

        # Calculate thermal noise spectral density and per-RB noise power
        self.n0 = self.BOLTZMANN_K * self.noise_temp_k  # W/Hz
        self.noise_power_per_rb = self.n0 * self.rb_bandwidth_hz  # Watts per RB

        # Precompute FSPL constant: 20*log10(4*pi*f/c)
        self.fspl_const_db = 20.0 * np.log10(4.0 * np.pi * self.carrier_freq_hz / self.C_LIGHT)

    def compute_elevation_angle(self, sat_pos_km: np.ndarray, cell_pos_km: np.ndarray) -> float:
        """
        Compute the elevation angle (in degrees) from a ground cell to a satellite.

        Args:
            sat_pos_km: [x, y, z] satellite position in km.
            cell_pos_km: [x, y, z] ground cell position in km.

        Returns:
            Elevation angle in degrees in [-90, 90].
        """
        rel_vector = sat_pos_km - cell_pos_km
        distance = np.linalg.norm(rel_vector)
        if distance < 1e-6:
            return 90.0

        # Zenith unit vector pointing away from Earth center at ground cell
        cell_norm = np.linalg.norm(cell_pos_km)
        if cell_norm < 1e-6:
            zenith = np.array([0.0, 0.0, 1.0])
        else:
            zenith = cell_pos_km / cell_norm

        sin_elevation = np.dot(rel_vector, zenith) / distance
        sin_elevation = float(np.clip(sin_elevation, -1.0, 1.0))
        elevation_rad = np.arcsin(sin_elevation)
        return float(np.degrees(elevation_rad))

    def compute_slant_range(self, sat_pos_km: np.ndarray, cell_pos_km: np.ndarray) -> float:
        """Slant range distance in km."""
        return float(np.linalg.norm(sat_pos_km - cell_pos_km))

    def compute_propagation_delay(self, slant_range_km: float) -> float:
        """Propagation delay in seconds."""
        slant_range_m = slant_range_km * 1000.0
        return float(slant_range_m / self.C_LIGHT)

    def is_visible(self, sat_pos_km: np.ndarray, cell_pos_km: np.ndarray) -> bool:
        """Check if satellite satisfies minimum elevation angle constraint."""
        elev = self.compute_elevation_angle(sat_pos_km, cell_pos_km)
        return elev >= self.min_elevation_deg

    def compute_fspl_db(self, slant_range_km: float) -> float:
        """
        Free-space path loss in dB.
        PL_FSPL = 20*log10(d_m) + 20*log10(f_Hz) + 20*log10(4*pi/c)
        """
        slant_range_m = max(1.0, slant_range_km * 1000.0)
        return float(20.0 * np.log10(slant_range_m) + self.fspl_const_db)

    def compute_atmospheric_loss_db(self, elevation_deg: float) -> float:
        """
        Elevation-dependent atmospheric attenuation in dB.
        Simplified ITU-R model: zenith absorption / sin(elevation) + nominal rain/cloud margin.
        """
        eff_elev = max(5.0, elevation_deg)
        elev_rad = np.radians(eff_elev)
        zenith_loss_db = 0.35  # Gas absorption at 20 GHz
        rain_cloud_margin_db = 0.45
        return float((zenith_loss_db / np.sin(elev_rad)) + rain_cloud_margin_db)

    def compute_transmit_antenna_gain(
        self,
        sat_pos_km: np.ndarray,
        intended_cell_pos_km: np.ndarray,
        target_cell_pos_km: np.ndarray,
    ) -> float:
        """
        Compute satellite transmit antenna gain (in dBi) toward target_cell_pos_km
        when the beam is steered toward intended_cell_pos_km.

        If target == intended, returns peak antenna gain G_tx,max.
        Otherwise, computes off-axis angle and applies parabolic roll-off pattern.
        """
        v_intended = intended_cell_pos_km - sat_pos_km
        d_intended = np.linalg.norm(v_intended)

        v_target = target_cell_pos_km - sat_pos_km
        d_target = np.linalg.norm(v_target)

        if d_intended < 1e-6 or d_target < 1e-6:
            return self.tx_antenna_gain_dbi

        cos_angle = np.dot(v_intended, v_target) / (d_intended * d_target)
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        off_axis_deg = np.degrees(np.arccos(cos_angle))

        # Standard ITU-R antenna roll-off: G(theta) = G_max - 12 * (theta / theta_3dB)^2
        roll_off = 12.0 * (off_axis_deg / self.beam_3db_width_deg) ** 2
        gain_dbi = max(self.min_sidelobe_gain_dbi, self.tx_antenna_gain_dbi - roll_off)
        return float(gain_dbi)

    def compute_channel_gain(
        self,
        sat_pos_km: np.ndarray,
        cell_pos_km: np.ndarray,
        steered_cell_pos_km: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate total linear power gain h_i,c between satellite i and ground cell c.

        Args:
            sat_pos_km: Satellite position [x, y, z] in km.
            cell_pos_km: Ground cell receiver position [x, y, z] in km.
            steered_cell_pos_km: If beam is steered at a different cell, calculates off-axis gain.
                                If None, assumes beam steered directly at cell_pos_km (main-lobe).

        Returns:
            Linear power gain h_i,c (dimensionless ratio, <= 1.0).
        """
        elev_deg = self.compute_elevation_angle(sat_pos_km, cell_pos_km)
        if elev_deg < self.min_elevation_deg:
            # Below visibility mask: highly attenuated / blocked
            return 1e-25

        slant_range_km = self.compute_slant_range(sat_pos_km, cell_pos_km)
        fspl_db = self.compute_fspl_db(slant_range_km)
        atm_db = self.compute_atmospheric_loss_db(elev_deg)

        if steered_cell_pos_km is None:
            tx_gain_dbi = self.tx_antenna_gain_dbi
        else:
            tx_gain_dbi = self.compute_transmit_antenna_gain(
                sat_pos_km=sat_pos_km,
                intended_cell_pos_km=steered_cell_pos_km,
                target_cell_pos_km=cell_pos_km,
            )

        rx_gain_dbi = self.rx_antenna_gain_dbi

        total_gain_db = tx_gain_dbi + rx_gain_dbi - fspl_db - atm_db
        # Convert dB to linear power ratio
        h_linear = 10.0 ** (total_gain_db / 10.0)
        return float(h_linear)

    def compute_sinr(
        self,
        rx_signal_power_watts: float,
        interference_power_watts: float,
        num_rbs: int = 1,
    ) -> float:
        """
        Compute SINR = P_rx / (N_0 * B + I).

        Args:
            rx_signal_power_watts: Desired received signal power (Watts).
            interference_power_watts: Aggregate co-channel interference power (Watts).
            num_rbs: Number of occupied resource blocks.

        Returns:
            Linear SINR ratio.
        """
        total_noise_power = self.noise_power_per_rb * max(1, num_rbs)
        denominator = total_noise_power + max(0.0, interference_power_watts)
        if denominator <= 0.0 or rx_signal_power_watts <= 0.0:
            return 0.0
        return float(rx_signal_power_watts / denominator)

    def compute_achievable_rate_mbps(self, sinr_linear: float, num_rbs: int = 1) -> float:
        """
        Shannon data rate with implementation efficiency:
            R = eta * B * log2(1 + SINR)

        Args:
            sinr_linear: Linear SINR.
            num_rbs: Number of allocated resource blocks.

        Returns:
            Data rate in Mbps (Megabits per second).
        """
        if sinr_linear <= 0.0 or num_rbs <= 0:
            return 0.0
        allocated_bw_hz = self.rb_bandwidth_hz * num_rbs
        shannon_bps = self.efficiency * allocated_bw_hz * np.log2(1.0 + sinr_linear)
        return float(shannon_bps / 1e6)  # Convert bps to Mbps

