"""Trajectory collection and counterfactual credit calculation."""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np

from environment.leo_env import MultiShellLEOEnv
from agents.mappo import ShellMeanMAPPO
from training.buffer import RolloutBuffer
from training.advantages import compute_gae_and_returns, normalize_advantages


def collect_rollout(
    env: MultiShellLEOEnv,
    agent: ShellMeanMAPPO,
    buffer: RolloutBuffer,
    use_counterfactual_credit: bool = True,
    current_obs: Optional[Dict[str, np.ndarray]] = None,
    current_info: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, float]]:
    """
    Collect a rollout of transitions and compute GAE advantages.

    Args:
        env: MultiShellLEOEnv instance.
        agent: ShellMeanMAPPO agent instance.
        buffer: RolloutBuffer to store transitions.
        use_counterfactual_credit: Whether to calculate marginal replay credit.
        current_obs: Current observations from prior step/reset.
        current_info: Current info dict from prior step/reset.

    Returns:
        Tuple of (last_observations, last_info, rollout_telemetry).
    """
    buffer.reset()

    if current_obs is None or current_info is None:
        obs_dict, info = env.reset()
    else:
        obs_dict, info = current_obs, current_info

    rollout_rewards = []
    rollout_served = []
    rollout_retention = []

    while not buffer.is_full():
        masks_dict = env.get_action_masks()
        global_state = info["global_state"]

        # Compute value estimate V(S_t)
        value = agent.get_value(global_state)

        # Generate policy actions
        actions, log_probs, beam_log_probs, action_arrays = agent.get_actions(
            observations=obs_dict,
            action_masks=masks_dict,
            deterministic=False,
        )

        # Prepare masks for buffer storage
        cell_masks = {}
        rb_masks = {}
        for s_idx in range(agent.actor.num_satellites):
            sat_key = f"sat_{s_idx}"
            sat_vis = masks_dict[sat_key]["visible_cells"]
            c_mask = agent.mask_mgr.get_cell_mask(sat_vis)
            cell_masks[sat_key] = c_mask

            # Per-beam RB masks based on allocated RBs
            r_mask_beams = []
            occupied: set[int] = set()
            for b_idx in range(agent.beams_per_satellite):
                r_mask = agent.mask_mgr.get_rb_mask(occupied)
                r_mask_beams.append(r_mask)
                # If active, add occupied
                c_val = action_arrays[sat_key]["cell_actions"][b_idx]
                r_val = action_arrays[sat_key]["rb_actions"][b_idx]
                if c_val > 0:
                    s_rb, w_rb = agent.mask_mgr.decode_rb_action(r_val)
                    occupied.update(range(s_rb, s_rb + w_rb))
            rb_masks[sat_key] = np.array(r_mask_beams, dtype=bool)

        # Step in environment
        next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated

        # Counterfactual beam credit estimation
        beam_credits = {}
        if use_counterfactual_credit:
            team_u = float(next_info["team_reward"])
            for s_idx in range(agent.actor.num_satellites):
                sat_key = f"sat_{s_idx}"
                sat_credits = np.zeros(agent.beams_per_satellite, dtype=np.float32)
                for b_idx in range(agent.beams_per_satellite):
                    c_act = action_arrays[sat_key]["cell_actions"][b_idx]
                    if c_act > 0:
                        # Active beam: evaluate counterfactual utility with beam turned off
                        cf_u = env.evaluate_counterfactual_step(actions, target_sat_id=s_idx, target_beam_id=b_idx)
                        sat_credits[b_idx] = float(team_u - cf_u)
                    else:
                        sat_credits[b_idx] = 0.0
                beam_credits[sat_key] = sat_credits

        # Insert transition into buffer
        buffer.insert(
            obs_dict=obs_dict,
            global_state=global_state,
            actions_dict=actions,
            action_arrays_dict=action_arrays,
            cell_masks_dict=cell_masks,
            rb_masks_dict=rb_masks,
            log_probs_dict=log_probs,
            beam_log_probs_dict=beam_log_probs,
            beam_credits_dict=beam_credits if use_counterfactual_credit else None,
            rewards_dict=rewards,
            value=value,
            done=done,
        )

        rollout_rewards.append(next_info["team_reward"])
        rollout_served.append(next_info["served_traffic_mb"])
        num_sub = next_info["num_retained"] + next_info["num_rejected"]
        ret_rate = (next_info["num_retained"] / num_sub * 100.0) if num_sub > 0 else 100.0
        rollout_retention.append(ret_rate)

        obs_dict = next_obs
        info = next_info

        if done:
            obs_dict, info = env.reset()

    # Compute GAE advantages and returns across the filled buffer
    last_val = agent.get_value(info["global_state"])
    team_rewards = np.mean(buffer.rewards, axis=1)  # Team reward per slot

    gamma = float(agent.config.get("mappo", {}).get("gamma", 0.99))
    gae_lambda = float(agent.config.get("mappo", {}).get("gae_lambda", 0.95))

    advantages, returns = compute_gae_and_returns(
        rewards=team_rewards,
        values=buffer.values,
        dones=buffer.dones,
        last_value=last_val,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    buffer.advantages = advantages
    buffer.returns = returns

    telemetry = {
        "mean_reward": float(np.mean(rollout_rewards)),
        "mean_served_mb": float(np.mean(rollout_served)),
        "mean_retention_rate": float(np.mean(rollout_retention)),
    }

    return obs_dict, info, telemetry

