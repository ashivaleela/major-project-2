"""Multi-Agent Gymnasium-compatible simulation environment for multi-shell LEO downlink allocation."""

from __future__ import annotations
import copy
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.traffic_model import TrafficModel
from environment.channel_model import ChannelModel
from environment.satellite_network import SatelliteNetwork
from environment.resource_manager import ResourceManager, SatelliteAllocation
from environment.conflict_resolver import ConflictResolver, ServiceRequest, ResolutionOutcome
from environment.transmission_model import TransmissionModel, TransmissionResult


class MultiShellLEOEnv(gym.Env):
    """
    Simulation environment for downlink resource allocation in multi-shell LEO satellite networks.
    Features:
    - Multi-satellite, multi-beam downlink coordination across multiple orbital shells.
    - Deterministic conflict resolution for competing beam-cell requests.
    - Physical path loss, off-axis beam pattern attenuation, thermal noise, and co-channel interference.
    - Autoregressive action structure and counterfactual credit calculation capability.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the environment.

        Args:
            config: Full configuration dictionary.
            seed: Optional integer seed for reproducibility.
        """
        super().__init__()
        self.config = config
        self.seed_val = seed if seed is not None else config.get("experiment", {}).get("seed", 42)
        self.rng = np.random.default_rng(self.seed_val)

        net_cfg = config.get("network", {})
        rew_cfg = config.get("rewards", {})

        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.num_shells = int(net_cfg.get("num_shells", 2))
        self.sats_per_shell = list(net_cfg.get("satellites_per_shell", [4, 4]))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_ground_cells = int(net_cfg.get("num_ground_cells", 20))
        self.num_rbs = int(net_cfg.get("num_resource_blocks", 8))
        self.episode_length = int(net_cfg.get("episode_length", 100))
        self.time_step = float(net_cfg.get("time_step_sec", 1.0))

        # Reward weights
        self.w_thr = float(rew_cfg.get("w_throughput", 1.0))
        self.w_queue = float(rew_cfg.get("w_queue", 0.05))
        self.w_interf = float(rew_cfg.get("w_interference", 0.02))
        self.w_conflict = float(rew_cfg.get("w_conflict", 0.5))
        self.w_switch = float(rew_cfg.get("w_switching", 0.1))
        self.w_fair = float(rew_cfg.get("w_fairness", 2.0))

        # Core subsystems
        self.channel_model = ChannelModel(config)
        self.traffic_model = TrafficModel(config, rng=self.rng)
        self.satellite_network = SatelliteNetwork(config, channel_model=self.channel_model, rng=self.rng)
        self.resource_manager = ResourceManager(config)
        self.conflict_resolver = ConflictResolver(config)
        self.transmission_model = TransmissionModel(config, channel_model=self.channel_model)

        # Agent IDs
        self.agent_ids = [f"sat_{i}" for i in range(self.num_satellites)]
        self.current_step = 0

        # State tracking
        self.last_transmission_result: Optional[TransmissionResult] = None
        self.last_resolution_outcome: Optional[ResolutionOutcome] = None
        self.shell_metrics: Dict[int, Dict[str, float]] = {}

        # Observation dimension calculation:
        # Local sat features: (sat_id onehot: N_sat) + (shell_id onehot: N_shells) + pos(3) + vel(3) + prev_beams(K) = N_sat + N_shells + 6 + K
        # Ground cells: for each of 20 cells -> (backlog, visible_flag, channel_gain, elevation) = 4 * N_cells
        # Shell summaries: for each of N_shells -> (mean_q, active_ratio, power_util, rb_occ, interf_lvl) = 5 * N_shells
        self.local_feat_dim = self.num_satellites + self.num_shells + 6 + self.beams_per_satellite
        self.cells_feat_dim = 4 * self.num_ground_cells
        self.shell_sum_dim = 5 * self.num_shells
        self.obs_dim = self.local_feat_dim + self.cells_feat_dim + self.shell_sum_dim

        # Global critic state dim:
        # All sat features (N_sat * local_feat_dim) + ground cells state (2 * N_cells) + all shell summaries
        self.global_state_dim = (self.num_satellites * self.local_feat_dim) + (2 * self.num_ground_cells) + self.shell_sum_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Dict(
            {
                f"beam_{k}": spaces.Dict(
                    {
                        "cell_id": spaces.Discrete(self.num_ground_cells + 1),  # 0: inactive, 1..N: cell
                        "rb_start": spaces.Discrete(self.num_rbs),
                        "rb_width": spaces.Discrete(self.num_rbs + 1),
                        "power_share": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    }
                )
                for k in range(self.beams_per_satellite)
            }
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment for a new episode.

        Returns:
            Tuple of (observations_dict, info_dict).
        """
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(seed)
            self.traffic_model.rng = self.rng
            self.satellite_network.rng = self.rng

        self.current_step = 0
        self.traffic_model.reset()
        self.satellite_network.reset()

        # Update initial visibility and channel gains
        cell_positions = self.traffic_model.get_positions()
        self.satellite_network.update_visibility_and_channels(cell_positions)

        # Clear tracking
        self.last_transmission_result = None
        self.last_resolution_outcome = None
        self._update_shell_summaries(None)

        observations = self._get_all_observations()
        info = {
            "step": self.current_step,
            "queues": self.traffic_model.get_queues().copy(),
            "global_state": self.get_global_state(),
        }
        return observations, info

    def _update_shell_summaries(self, trans_result: Optional[TransmissionResult]) -> None:
        """Compute compact per-shell aggregate statistics."""
        queues = self.traffic_model.get_queues()
        max_q = self.traffic_model.max_capacity

        for shell_idx in range(self.num_shells):
            shell_sats = self.satellite_network.get_shell_satellites(shell_idx)
            num_sats = len(shell_sats)
            if num_sats == 0:
                self.shell_metrics[shell_idx] = {
                    "mean_queue": 0.0,
                    "active_ratio": 0.0,
                    "power_util": 0.0,
                    "rb_occupancy": 0.0,
                    "interf_level": 0.0,
                }
                continue

            # Visible cells in this shell
            shell_vis_cells: set[int] = set()
            for sat in shell_sats:
                shell_vis_cells.update(sat.visible_cells)

            if shell_vis_cells:
                mean_q = float(np.mean([queues[c] for c in shell_vis_cells])) / max_q
            else:
                mean_q = 0.0

            active_beams_count = 0
            total_beams = num_sats * self.beams_per_satellite
            total_power_util = 0.0
            total_occupied_rbs = 0
            total_interf = 0.0

            if trans_result is not None:
                for sat in shell_sats:
                    for b_idx in range(self.beams_per_satellite):
                        bm = trans_result.beam_metrics.get((sat.sat_id, b_idx))
                        if bm and bm.is_retained and bm.is_active:
                            active_beams_count += 1
                            total_power_util += bm.tx_power_w / sat.max_tx_power_w
                            total_occupied_rbs += bm.rb_width
                            total_interf += bm.interference_w

            active_ratio = active_beams_count / max(1, total_beams)
            power_util = total_power_util / max(1, num_sats)
            rb_occ = total_occupied_rbs / max(1, num_sats * self.num_rbs)
            interf_lvl = float(np.tanh(total_interf * 1e11))  # Normalized interference metric

            self.shell_metrics[shell_idx] = {
                "mean_queue": float(mean_q),
                "active_ratio": float(active_ratio),
                "power_util": float(power_util),
                "rb_occupancy": float(rb_occ),
                "interf_level": float(interf_lvl),
            }

    def _get_observation(self, sat_id: int) -> np.ndarray:
        """Construct normalized observation vector for satellite sat_id."""
        sat = self.satellite_network.satellites[sat_id]
        queues = self.traffic_model.get_queues()
        max_q = self.traffic_model.max_capacity

        # 1. Local satellite features
        sat_id_onehot = np.zeros(self.num_satellites, dtype=np.float32)
        sat_id_onehot[sat_id] = 1.0

        shell_id_onehot = np.zeros(self.num_shells, dtype=np.float32)
        shell_id_onehot[sat.shell_id] = 1.0

        norm_pos = (sat.position / 10000.0).astype(np.float32)
        norm_vel = (sat.velocity / 10.0).astype(np.float32)
        prev_cells = np.array(
            [c / float(self.num_ground_cells) for c in sat.previous_beam_cell], dtype=np.float32
        )

        local_feats = np.concatenate([sat_id_onehot, shell_id_onehot, norm_pos, norm_vel, prev_cells])

        # 2. Ground cells features
        cell_feats_list = []
        for c in range(self.num_ground_cells):
            q_norm = float(queues[c] / max_q)
            is_vis = 1.0 if c in sat.visible_cells else 0.0
            h_gain = float(sat.channel_gains[c]) if c in sat.visible_cells else 0.0
            # Normalize channel gain (typical LEO gain is ~1e-12 to 1e-10)
            h_norm = float(np.clip(h_gain * 1e11, 0.0, 10.0))
            elev = self.channel_model.compute_elevation_angle(sat.position, self.traffic_model.cells[c].position)
            elev_norm = float(np.clip(elev / 90.0, -1.0, 1.0))
            cell_feats_list.extend([q_norm, is_vis, h_norm, elev_norm])

        cell_feats = np.array(cell_feats_list, dtype=np.float32)

        # 3. Shell summaries
        shell_feats_list = []
        for s in range(self.num_shells):
            sm = self.shell_metrics.get(s, {})
            shell_feats_list.extend(
                [
                    sm.get("mean_queue", 0.0),
                    sm.get("active_ratio", 0.0),
                    sm.get("power_util", 0.0),
                    sm.get("rb_occupancy", 0.0),
                    sm.get("interf_level", 0.0),
                ]
            )
        shell_feats = np.array(shell_feats_list, dtype=np.float32)

        obs = np.concatenate([local_feats, cell_feats, shell_feats])
        return obs.astype(np.float32)

    def _get_all_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all satellites."""
        return {f"sat_{i}": self._get_observation(i) for i in range(self.num_satellites)}

    def get_global_state(self) -> np.ndarray:
        """Construct centralized critic state vector."""
        queues = self.traffic_model.get_queues()
        max_q = self.traffic_model.max_capacity

        all_local = []
        for i in range(self.num_satellites):
            sat = self.satellite_network.satellites[i]
            sat_id_onehot = np.zeros(self.num_satellites, dtype=np.float32)
            sat_id_onehot[i] = 1.0
            shell_onehot = np.zeros(self.num_shells, dtype=np.float32)
            shell_onehot[sat.shell_id] = 1.0
            norm_pos = (sat.position / 10000.0).astype(np.float32)
            norm_vel = (sat.velocity / 10.0).astype(np.float32)
            prev_cells = np.array([c / float(self.num_ground_cells) for c in sat.previous_beam_cell], dtype=np.float32)
            all_local.append(np.concatenate([sat_id_onehot, shell_onehot, norm_pos, norm_vel, prev_cells]))

        flat_locals = np.concatenate(all_local)

        # Global cell queues and hotspot flags
        hotspot_flags = self.traffic_model.get_hotspot_flags().astype(np.float32)
        norm_queues = (queues / max_q).astype(np.float32)
        cell_state = np.concatenate([norm_queues, hotspot_flags])

        # Shell summaries
        shell_feats = []
        for s in range(self.num_shells):
            sm = self.shell_metrics.get(s, {})
            shell_feats.extend(
                [
                    sm.get("mean_queue", 0.0),
                    sm.get("active_ratio", 0.0),
                    sm.get("power_util", 0.0),
                    sm.get("rb_occupancy", 0.0),
                    sm.get("interf_level", 0.0),
                ]
            )

        global_vec = np.concatenate([flat_locals, cell_state, np.array(shell_feats, dtype=np.float32)])
        return global_vec.astype(np.float32)

    def get_action_masks(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate action masks for each satellite agent.
        Precludes invisible/infeasible ground cells from selection.
        """
        masks = {}
        for i in range(self.num_satellites):
            sat = self.satellite_network.satellites[i]
            # Cell mask: length = num_ground_cells + 1. Index 0 (inactive) always valid (1)
            cell_mask = np.zeros(self.num_ground_cells + 1, dtype=bool)
            cell_mask[0] = True
            for c in sat.visible_cells:
                cell_mask[c + 1] = True

            masks[f"sat_{i}"] = {
                "cell_mask": cell_mask,
                "visible_cells": list(sat.visible_cells),
            }
        return masks

    def step(
        self,
        actions: Dict[str, Any],
    ) -> tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Execute one full multi-agent scheduling slot.

        Args:
            actions: Dict mapping agent_id ('sat_0', ...) to action dict or structured allocation.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info).
        """
        self.current_step += 1

        cell_positions = self.traffic_model.get_positions()
        visibility_matrix = self.satellite_network.update_visibility_and_channels(cell_positions)
        current_queues = self.traffic_model.get_queues()

        # Step 1: Validate and sanitize actions via ResourceManager
        submitted_requests: List[ServiceRequest] = []
        allocations_by_sat: Dict[int, SatelliteAllocation] = {}
        switching_count = 0

        for sat_id in range(self.num_satellites):
            sat_key = f"sat_{sat_id}"
            sat = self.satellite_network.satellites[sat_id]
            raw_act = actions.get(sat_key, {})

            # Parse action format
            raw_beam_list = []
            if isinstance(raw_act, dict):
                for b_idx in range(self.beams_per_satellite):
                    b_dict = raw_act.get(f"beam_{b_idx}", {})
                    raw_beam_list.append(b_dict)
            elif isinstance(raw_act, list):
                raw_beam_list = raw_act

            alloc = self.resource_manager.validate_action(
                sat_id=sat_id,
                raw_beam_actions=raw_beam_list,
                visible_cell_indices=sat.visible_cells,
            )
            allocations_by_sat[sat_id] = alloc

            # Check beam retargeting / switching
            for b_idx, beam_alloc in enumerate(alloc.beam_allocations):
                prev_c = sat.previous_beam_cell[b_idx]
                curr_c = beam_alloc.cell_id
                if curr_c > 0 and curr_c != prev_c:
                    switching_count += 1

                # Generate ServiceRequest
                c_idx = beam_alloc.target_cell_index
                ch_gain = sat.channel_gains[c_idx] if c_idx is not None else 0.0
                slant_km = self.channel_model.compute_slant_range(sat.position, cell_positions[c_idx]) if c_idx is not None else 1000.0
                delay_sec = self.channel_model.compute_propagation_delay(slant_km)

                req = ServiceRequest(
                    sat_id=sat_id,
                    shell_id=sat.shell_id,
                    beam_id=b_idx,
                    cell_id=beam_alloc.cell_id,
                    power_share=beam_alloc.power_share,
                    rb_start=beam_alloc.rb_start,
                    rb_width=beam_alloc.rb_width,
                    channel_gain=ch_gain,
                    propagation_delay_sec=delay_sec,
                )
                submitted_requests.append(req)

        # Step 2: Conflict Resolution
        outcome = self.conflict_resolver.resolve_conflicts(
            requests=submitted_requests,
            queue_backlogs=current_queues,
            visibility_matrix=visibility_matrix,
        )
        self.last_resolution_outcome = outcome

        # Step 3: Physical Downlink Transmission & Interference
        sat_positions = {s.sat_id: s.position for s in self.satellite_network.satellites}
        trans_result = self.transmission_model.execute_transmissions(
            retained_requests=outcome.retained_requests,
            rejected_requests=outcome.rejected_requests,
            satellite_positions_km=sat_positions,
            cell_positions_km=cell_positions,
            queue_backlogs_mb=current_queues,
        )
        self.last_transmission_result = trans_result

        # Step 4: Traffic Queue Update
        new_queues, effective_served, arrivals = self.traffic_model.step(trans_result.cell_served_mb)

        # Step 5: Advance Orbits
        self.satellite_network.step_orbits()
        self.satellite_network.update_visibility_and_channels(cell_positions)

        # Update previous beam targets on satellites
        for sat_id, alloc in allocations_by_sat.items():
            sat = self.satellite_network.satellites[sat_id]
            for b_idx, beam_alloc in enumerate(alloc.beam_allocations):
                # Only update previous target if beam was retained and transmitted
                key = (sat_id, b_idx)
                if key in trans_result.beam_metrics and trans_result.beam_metrics[key].is_retained:
                    sat.previous_beam_cell[b_idx] = beam_alloc.cell_id
                else:
                    sat.previous_beam_cell[b_idx] = 0

        # Step 6: Update Per-Shell Summaries
        self._update_shell_summaries(trans_result)

        # Step 7: Compute Rewards
        # Calculate Jain's fairness index across served traffic
        sum_served = float(np.sum(effective_served))
        sum_served_sq = float(np.sum(effective_served**2))
        if sum_served_sq > 0.0:
            jains_fairness = (sum_served**2) / (self.num_ground_cells * sum_served_sq)
        else:
            jains_fairness = 1.0

        num_rejected = len(outcome.rejected_requests) - sum(
            1 for r in outcome.rejected_requests if r.cell_id == 0
        )
        total_backlog = float(np.sum(new_queues))

        # Reward components
        r_thr = self.w_thr * sum_served
        r_queue = -self.w_queue * (total_backlog / float(self.num_ground_cells))
        r_interf = -self.w_interf * float(np.tanh(trans_result.total_interference_w * 1e11))
        r_conflict = -self.w_conflict * float(num_rejected)
        r_switch = -self.w_switch * float(switching_count)
        r_fair = self.w_fair * float(jains_fairness)

        team_reward = float(r_thr + r_queue + r_interf + r_conflict + r_switch + r_fair)
        rewards = {f"sat_{i}": team_reward for i in range(self.num_satellites)}

        # Episode termination
        terminated = False
        truncated = self.current_step >= self.episode_length

        observations = self._get_all_observations()

        info = {
            "step": self.current_step,
            "team_reward": team_reward,
            "served_traffic_mb": sum_served,
            "arrivals_mb": float(np.sum(arrivals)),
            "remaining_backlog_mb": total_backlog,
            "num_retained": len(outcome.retained_requests),
            "num_rejected": num_rejected,
            "interference_watts": trans_result.total_interference_w,
            "jains_fairness": jains_fairness,
            "switching_count": switching_count,
            "mean_sinr_db": trans_result.mean_sinr_db,
            "global_state": self.get_global_state(),
        }

        return observations, rewards, terminated, truncated, info

    def evaluate_counterfactual_step(
        self,
        base_actions: Dict[str, Any],
        target_sat_id: int,
        target_beam_id: int,
    ) -> float:
        """
        Replay the exact current slot with target beam (target_sat_id, target_beam_id) set to INACTIVE.
        Power from the deactivated beam is moved to reserved power.
        Keeps all other actions and current environmental conditions fixed.
        Does not mutate the actual environment state or random generator.

        Returns:
            Counterfactual system utility (reward) U(a_{-i,k}).
        """
        cell_positions = self.traffic_model.get_positions()
        current_queues = self.traffic_model.get_queues()
        # Non-mutating visibility check
        visibility_matrix = np.zeros((self.num_satellites, self.num_ground_cells), dtype=bool)
        for sat in self.satellite_network.satellites:
            for c in range(self.num_ground_cells):
                if c in sat.visible_cells:
                    visibility_matrix[sat.sat_id, c] = True

        submitted_requests: List[ServiceRequest] = []
        switching_count = 0

        for sat_id in range(self.num_satellites):
            sat_key = f"sat_{sat_id}"
            sat = self.satellite_network.satellites[sat_id]
            raw_act = base_actions.get(sat_key, {})

            raw_beam_list = []
            if isinstance(raw_act, dict):
                for b_idx in range(self.beams_per_satellite):
                    b_dict = copy.deepcopy(raw_act.get(f"beam_{b_idx}", {}))
                    if sat_id == target_sat_id and b_idx == target_beam_id:
                        # Deactivate target beam
                        b_dict["cell_id"] = 0
                        b_dict["rb_start"] = 0
                        b_dict["rb_width"] = 0
                        b_dict["power_share"] = 0.0
                    raw_beam_list.append(b_dict)
            elif isinstance(raw_act, list):
                raw_beam_list = copy.deepcopy(raw_act)
                if sat_id == target_sat_id and target_beam_id < len(raw_beam_list):
                    raw_beam_list[target_beam_id]["cell_id"] = 0
                    raw_beam_list[target_beam_id]["rb_start"] = 0
                    raw_beam_list[target_beam_id]["rb_width"] = 0
                    raw_beam_list[target_beam_id]["power_share"] = 0.0

            alloc = self.resource_manager.validate_action(
                sat_id=sat_id,
                raw_beam_actions=raw_beam_list,
                visible_cell_indices=sat.visible_cells,
            )

            for b_idx, beam_alloc in enumerate(alloc.beam_allocations):
                prev_c = sat.previous_beam_cell[b_idx]
                curr_c = beam_alloc.cell_id
                if curr_c > 0 and curr_c != prev_c:
                    switching_count += 1

                c_idx = beam_alloc.target_cell_index
                ch_gain = sat.channel_gains[c_idx] if c_idx is not None else 0.0
                slant_km = (
                    self.channel_model.compute_slant_range(sat.position, cell_positions[c_idx])
                    if c_idx is not None
                    else 1000.0
                )
                delay_sec = self.channel_model.compute_propagation_delay(slant_km)

                req = ServiceRequest(
                    sat_id=sat_id,
                    shell_id=sat.shell_id,
                    beam_id=b_idx,
                    cell_id=beam_alloc.cell_id,
                    power_share=beam_alloc.power_share,
                    rb_start=beam_alloc.rb_start,
                    rb_width=beam_alloc.rb_width,
                    channel_gain=ch_gain,
                    propagation_delay_sec=delay_sec,
                )
                submitted_requests.append(req)

        # Conflict resolution
        outcome = self.conflict_resolver.resolve_conflicts(
            requests=submitted_requests,
            queue_backlogs=current_queues,
            visibility_matrix=visibility_matrix,
        )

        # Transmission simulation
        sat_positions = {s.sat_id: s.position for s in self.satellite_network.satellites}
        trans_result = self.transmission_model.execute_transmissions(
            retained_requests=outcome.retained_requests,
            rejected_requests=outcome.rejected_requests,
            satellite_positions_km=sat_positions,
            cell_positions_km=cell_positions,
            queue_backlogs_mb=current_queues,
        )

        # Counterfactual service & queues without mutating real traffic model
        served_traffic = trans_result.cell_served_mb
        post_service_queues = np.maximum(0.0, current_queues - served_traffic)

        sum_served = float(np.sum(served_traffic))
        sum_served_sq = float(np.sum(served_traffic**2))
        if sum_served_sq > 0.0:
            jains_fairness = (sum_served**2) / (self.num_ground_cells * sum_served_sq)
        else:
            jains_fairness = 1.0

        num_rejected = len(outcome.rejected_requests) - sum(
            1 for r in outcome.rejected_requests if r.cell_id == 0
        )
        total_backlog = float(np.sum(post_service_queues))

        r_thr = self.w_thr * sum_served
        r_queue = -self.w_queue * (total_backlog / float(self.num_ground_cells))
        r_interf = -self.w_interf * float(np.tanh(trans_result.total_interference_w * 1e11))
        r_conflict = -self.w_conflict * float(num_rejected)
        r_switch = -self.w_switch * float(switching_count)
        r_fair = self.w_fair * float(jains_fairness)

        cf_utility = float(r_thr + r_queue + r_interf + r_conflict + r_switch + r_fair)
        return cf_utility

