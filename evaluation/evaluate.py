"""Policy evaluation pipeline for baselines and trained MAPPO models."""

from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from environment.leo_env import MultiShellLEOEnv
from evaluation.metrics import SlotMetrics, EpisodeSummary, MetricsTracker


def run_episode(
    env: MultiShellLEOEnv,
    policy: Any,
    seed: int,
) -> EpisodeSummary:
    """
    Run a single evaluation episode with full telemetry logging.

    Args:
        env: MultiShellLEOEnv instance.
        policy: Policy instance (RandomPolicy, GreedyPolicy, RuleBasedPolicy, or MAPPOTrainer).
        seed: Random seed for episode.

    Returns:
        EpisodeSummary containing slot-by-slot metrics and aggregated statistics.
    """
    observations, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed=seed)

    initial_backlog = float(np.sum(info["queues"]))
    slot_records: List[SlotMetrics] = []
    prev_backlog = initial_backlog

    done = False
    step_count = 0

    while not done:
        masks = env.get_action_masks()
        t_start = time.perf_counter()

        # Generate policy actions
        if hasattr(policy, "select_actions"):
            # Check if policy accepts env parameter
            try:
                actions = policy.select_actions(observations, masks, env=env)
            except TypeError:
                actions = policy.select_actions(observations, masks)
        elif hasattr(policy, "step_evaluation"):
            actions = policy.step_evaluation(observations, masks)
        else:
            raise ValueError(f"Unknown policy type: {type(policy)}")

        t_end = time.perf_counter()
        slot_runtime_ms = (t_end - t_start) * 1000.0

        next_obs, rewards, terminated, truncated, step_info = env.step(actions)
        done = terminated or truncated
        step_count += 1

        curr_backlog = float(step_info["remaining_backlog_mb"])
        growth = curr_backlog - prev_backlog
        prev_backlog = curr_backlog

        num_ret = int(step_info["num_retained"])
        num_rej = int(step_info["num_rejected"])
        total_sub = num_ret + num_rej
        ret_rate = (num_ret / total_sub * 100.0) if total_sub > 0 else 100.0
        conf_rate = (num_rej / total_sub * 100.0) if total_sub > 0 else 0.0

        # Resource utilizations
        tot_possible_beams = env.num_satellites * env.beams_per_satellite
        active_ratio = (num_ret / tot_possible_beams) if tot_possible_beams > 0 else 0.0

        # RB utilization
        occupied_rbs = 0
        if env.last_transmission_result:
            for bm in env.last_transmission_result.beam_metrics.values():
                if bm.is_retained and bm.is_active:
                    occupied_rbs += bm.rb_width
        tot_possible_rbs = env.num_satellites * env.num_rbs
        rb_util = (occupied_rbs / tot_possible_rbs * 100.0) if tot_possible_rbs > 0 else 0.0

        # Power utilization
        tot_power_w = 0.0
        if env.last_transmission_result:
            for bm in env.last_transmission_result.beam_metrics.values():
                if bm.is_retained and bm.is_active:
                    tot_power_w += bm.tx_power_w
        tot_possible_power = env.num_satellites * env.resource_manager.max_tx_power_w
        pwr_util = (tot_power_w / tot_possible_power * 100.0) if tot_possible_power > 0 else 0.0

        slot_m = SlotMetrics(
            step=step_count,
            team_reward=float(step_info["team_reward"]),
            served_traffic_mb=float(step_info["served_traffic_mb"]),
            throughput_mbps=float(step_info["served_traffic_mb"] / env.time_step),
            total_backlog_mb=curr_backlog,
            queue_growth_mb=growth,
            num_submitted=total_sub,
            num_retained=num_ret,
            num_rejected=num_rej,
            retention_rate=ret_rate,
            conflict_rate=conf_rate,
            interference_watts=float(step_info["interference_watts"]),
            jains_fairness=float(step_info["jains_fairness"]),
            active_beam_ratio=active_ratio,
            rb_utilization=rb_util,
            power_utilization=pwr_util,
            switching_count=int(step_info["switching_count"]),
            mean_sinr_db=float(step_info["mean_sinr_db"]),
            slot_runtime_ms=slot_runtime_ms,
        )
        slot_records.append(slot_m)
        observations = next_obs

    summary = MetricsTracker.aggregate_episode(slot_records, initial_backlog)
    return summary


def evaluate_policy(
    env: MultiShellLEOEnv,
    policy: Any,
    policy_name: str,
    seeds: List[int],
) -> Dict[str, Any]:
    """
    Evaluate a policy across multiple seeds.

    Args:
        env: MultiShellLEOEnv instance.
        policy: The policy to evaluate.
        policy_name: Name label for reporting.
        seeds: List of random seeds.

    Returns:
        Dict containing raw episode summaries and aggregated mean +/- std metrics.
    """
    episode_summaries: List[EpisodeSummary] = []
    records = []

    for seed in seeds:
        summary = run_episode(env, policy, seed=seed)
        episode_summaries.append(summary)
        row = summary.to_dict()
        row["seed"] = seed
        row["policy"] = policy_name
        records.append(row)

    df = pd.DataFrame(records)
    metric_cols = [c for c in df.columns if c not in ["seed", "policy"]]

    means = df[metric_cols].mean().to_dict()
    stds = df[metric_cols].std().fillna(0.0).to_dict()

    return {
        "policy": policy_name,
        "dataframe": df,
        "means": means,
        "stds": stds,
        "summaries": episode_summaries,
    }

