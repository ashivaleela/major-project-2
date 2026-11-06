"""Main Command-Line Interface (CLI) for ShellMean-MAPPO.

A Conflict-Aware Multi-Agent Reinforcement Learning Framework for Downlink
Resource Allocation in Multi-Shell LEO Satellite Networks.
"""

from __future__ import annotations
import os
import sys
import tempfile
import argparse
import yaml
import numpy as np
import pandas as pd

# Suppress Matplotlib cache warnings on restricted environments
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_shellmean"))

# Ensure clean UTF-8 console output for symbols on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from environment.leo_env import MultiShellLEOEnv
from training.train import MAPPOTrainer
from training.checkpoint import load_checkpoint
from baselines.random_policy import RandomPolicy
from baselines.greedy_policy import GreedyPolicy
from baselines.rule_based_policy import RuleBasedPolicy
from evaluation.evaluate import evaluate_policy, run_episode
from evaluation.compare_baselines import run_baseline_comparison
from evaluation.ablation import run_ablation_study
from evaluation.plots import (
    plot_training_curves,
    plot_baseline_comparison,
    plot_ablation_comparison,
)


def load_yaml_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mode_simulate(
    config: dict,
    episodes: int = 1,
    seed: int = 42,
    policy_name: str = "rule_based",
    checkpoint_path: str = "checkpoints/best_model.pt",
) -> None:
    """
    Run detailed multi-agent simulator demonstration showing complete scheduling slots.
    Outputs step-by-step physical breakdown and saves execution telemetry to log file.
    """
    net_cfg = config.get("network", {})
    orb_cfg = config.get("orbit", {})

    num_sats = int(net_cfg.get("num_satellites", 8))
    num_shells = int(net_cfg.get("num_shells", 2))
    sats_per_shell = list(net_cfg.get("satellites_per_shell", [4, 4]))
    beams_per_sat = int(net_cfg.get("beams_per_satellite", 2))
    num_cells = int(net_cfg.get("num_ground_cells", 20))
    num_rbs = int(net_cfg.get("num_resource_blocks", 8))
    rb_bw_mhz = float(net_cfg.get("rb_bandwidth_mhz", 25.0))
    slot_sec = float(net_cfg.get("time_step_sec", 1.0))
    altitudes = list(orb_cfg.get("shell_altitudes_km", [550.0, 1100.0]))
    total_beams = num_sats * beams_per_sat

    log_dir = config.get("experiment", {}).get("logs_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "simulation_execution.log")

    log_lines = []

    def emit(text: str = "") -> None:
        print(text)
        log_lines.append(text)

    env = MultiShellLEOEnv(config, seed=seed)

    # Initialize requested policy
    if policy_name in ["trained", "mappo"]:
        emit(f"[Policy Selection] Initializing Trained ShellMean-MAPPO Agent from: {checkpoint_path}")
        trainer = MAPPOTrainer(config)
        if os.path.exists(checkpoint_path):
            load_checkpoint(trainer.agent, checkpoint_path)
            emit(f"[*] Successfully loaded trained checkpoint weights.")
        else:
            emit(f"[!] Warning: Checkpoint {checkpoint_path} not found. Using initialized MAPPO agent.")
        policy = trainer
        pol_label = "Trained ShellMean-MAPPO Policy"
    elif policy_name == "greedy":
        policy = GreedyPolicy(config, seed=seed)
        pol_label = "Greedy Scheduling Policy"
    elif policy_name == "random":
        policy = RandomPolicy(config, seed=seed)
        pol_label = "Random Scheduling Policy"
    else:
        policy = RuleBasedPolicy(config, seed=seed)
        pol_label = "Rule-Based Conflict-Aware Policy"

    for ep in range(episodes):
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)
        tot_q_init = float(np.sum(info["queues"]))

        masks = env.get_action_masks()
        if hasattr(policy, "select_actions"):
            try:
                actions = policy.select_actions(obs, masks, env=env)
            except TypeError:
                actions = policy.select_actions(obs, masks)
        else:
            actions = policy.step_evaluation(obs, masks)

        next_obs, rewards, term, trunc, step_info = env.step(actions)
        outcome = env.last_resolution_outcome
        trans_res = env.last_transmission_result

        num_retained = len(outcome.retained_requests)
        num_rejected = step_info["num_rejected"]
        total_requests = num_retained + num_rejected

        # Count active satellites in slot 1
        active_sats = sum(
            1 for s_id in range(num_sats)
            if any(
                actions.get(f"sat_{s_id}", {}).get(f"beam_{b}", {}).get("cell_id", 0) > 0
                for b in range(beams_per_sat)
            )
        )

        occupied_rbs = sum(
            bm.rb_width for bm in trans_res.beam_metrics.values()
            if bm.is_retained and bm.is_active
        )
        rb_util = (occupied_rbs / max(1, num_sats * num_rbs)) * 100.0

        # Primary Output Block
        emit("\n" + "=" * 48)
        emit("========== SHELLMEAN-MAPPO SIMULATION ==========")
        emit("=" * 48)
        emit()
        emit("Network Configuration:")
        emit(f"Number of Satellites:      {num_sats} (Shell 0: {sats_per_shell[0]}, Shell 1: {sats_per_shell[1]})")
        emit(f"Number of Orbital Shells:  {num_shells} (Altitudes: {altitudes[0]} km, {altitudes[1]} km)")
        emit(f"Number of Ground Cells:    {num_cells} (Hotspot: {int(num_cells*0.25)}, Regular: {num_cells - int(num_cells*0.25)})")
        emit(f"Number of Beams:           {total_beams} ({beams_per_sat} per satellite)")
        emit(f"Number of Resource Blocks: {num_rbs} ({rb_bw_mhz:.1f} MHz each, {num_rbs*rb_bw_mhz:.1f} MHz total)")
        emit(f"Active Scheduling Policy:  {pol_label}")
        emit()
        emit("Simulation:")
        emit(f"Episode:                   {ep + 1} / {episodes} (Seed: {ep_seed})")
        emit(f"Scheduling Slot:           1 / {env.episode_length}")
        emit(f"Active Satellites:         {active_sats} / {num_sats}")
        emit(f"Generated Traffic:         {step_info['arrivals_mb']:.2f} Mb")
        emit(f"Beam Requests:             {total_requests} proposed")
        emit()
        emit("Conflict Resolution:")
        emit(f"Total Requests:            {total_requests}")
        emit(f"Retained Requests:         {num_retained}")
        emit(f"Rejected Requests:         {num_rejected}")
        emit()
        emit("Transmission:")
        emit(f"SINR:                      {step_info['mean_sinr_db']:.2f} dB (Min: {trans_res.min_sinr_db:.1f} dB, Max: {trans_res.max_sinr_db:.1f} dB)")
        emit(f"Interference:              {(step_info['interference_watts'] * 1e12):.2f} pW")
        emit(f"Successfully Served Traffic: {step_info['served_traffic_mb']:.2f} Mb")
        emit()
        emit("Network Metrics:")
        emit(f"Remaining Queue:           {step_info['remaining_backlog_mb']:.2f} Mb (Pre-Service: {tot_q_init:.2f} Mb)")
        emit(f"Total Throughput:          {(step_info['served_traffic_mb'] / slot_sec):.2f} Mbps")
        emit(f"Resource Utilization:      {rb_util:.2f}% (RB Occupancy)")
        emit(f"Average Reward:            {step_info['team_reward']:.2f}")
        emit("=" * 48)

        # Detailed Breakdown
        emit("\n[Detailed Physical Layer Telemetry - Slot 1]")
        emit("  Visible Cells and Proposed Requests:")
        for s_id in range(num_sats):
            sat = env.satellite_network.satellites[s_id]
            vis = masks[f"sat_{s_id}"]["visible_cells"]
            b_act = actions[f"sat_{s_id}"]
            b0_c = b_act["beam_0"]["cell_id"]
            b1_c = b_act["beam_1"]["cell_id"]
            b0_str = f"Cell {b0_c-1} (RBs [{b_act['beam_0']['rb_start']}..{b_act['beam_0']['rb_start']+b_act['beam_0']['rb_width']-1}], Pwr={b_act['beam_0']['power_share']:.2f})" if b0_c > 0 else "Inactive"
            b1_str = f"Cell {b1_c-1} (RBs [{b_act['beam_1']['rb_start']}..{b_act['beam_1']['rb_start']+b_act['beam_1']['rb_width']-1}], Pwr={b_act['beam_1']['power_share']:.2f})" if b1_c > 0 else "Inactive"
            emit(f"    Sat {s_id} (Shell {sat.shell_id}): Visible={vis} | Beam 0: {b0_str} | Beam 1: {b1_str}")

        emit("\n  Conflict Resolution Decisions:")
        if outcome.logs:
            for log_msg in outcome.logs[:6]:
                emit(f"    - {log_msg}")
            if len(outcome.logs) > 6:
                emit(f"    - ... and {len(outcome.logs) - 6} additional resolution decisions")

        emit("\n  Downlink Transmission Links:")
        for (sat_id, b_id), bm in trans_res.beam_metrics.items():
            if bm.is_retained and bm.is_active:
                c_idx = bm.target_cell_index
                emit(
                    f"    Beam (Sat {sat_id}, Beam {b_id}) -> Cell {c_idx}: "
                    f"RBs=[{bm.rb_start}..{bm.rb_start+bm.rb_width-1}] | "
                    f"P_tx={bm.tx_power_w:.1f}W | "
                    f"Interf={(bm.interference_w*1e12):.2f}pW | "
                    f"SINR={bm.sinr_db:.1f}dB | "
                    f"Rate={bm.rate_mbps:.1f}Mbps | "
                    f"Served={bm.served_data_mb:.2f}Mb"
                )

        # Simulate remaining slots of episode
        curr_obs = next_obs
        total_delivered = step_info["served_traffic_mb"]
        total_rewards = step_info["team_reward"]

        for step_idx in range(2, env.episode_length + 1):
            m = env.get_action_masks()
            if hasattr(policy, "select_actions"):
                try:
                    acts = policy.select_actions(curr_obs, m, env=env)
                except TypeError:
                    acts = policy.select_actions(curr_obs, m)
            else:
                acts = policy.step_evaluation(curr_obs, m)
            curr_obs, r, t, tr, s_info = env.step(acts)
            total_delivered += s_info["served_traffic_mb"]
            total_rewards += s_info["team_reward"]
            if t or tr:
                break

        emit("\n[Full Episode Summary]")
        emit(f"  * Cumulative Delivered Traffic: {total_delivered:.2f} Mb")
        emit(f"  * Mean Throughput:              {(total_delivered / env.episode_length):.2f} Mbps")
        emit(f"  * Cumulative Episode Reward:     {total_rewards:.2f}")
        emit(f"  * Final Remaining Backlog:       {float(np.sum(env.traffic_model.get_queues())):.2f} Mb")
        emit("=" * 48 + "\n")

    # Persist log
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Simulation execution log saved to: {log_file_path}")


def mode_train(config: dict, slots: int, checkpoint_path: str, eval_interval: int = 128) -> None:
    """Execute ShellMean-MAPPO training run."""
    print(f"\n=== Mode: Train ShellMean-MAPPO ({slots} slots) ===")
    trainer = MAPPOTrainer(config)
    df_history = trainer.train(
        total_slots=slots,
        eval_interval=eval_interval,
        checkpoint_path=checkpoint_path,
        use_counterfactual_credit=config.get("rewards", {}).get("use_counterfactual_credit", True),
    )

    log_path = os.path.join(config.get("experiment", {}).get("results_dir", "results"), "training_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df_history.to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path}")

    # Generate training curves plot
    plot_training_curves(df_history, output_path="results/training_curves.png")


def mode_evaluate(config: dict, checkpoint_path: str, seeds: list[int]) -> None:
    """Evaluate trained ShellMean-MAPPO model without policy updates."""
    print(f"\n=== Mode: Evaluate Checkpoint ({checkpoint_path}) ===")
    env = MultiShellLEOEnv(config, seed=seeds[0])
    trainer = MAPPOTrainer(config)

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        load_checkpoint(trainer.agent, checkpoint_path)
    else:
        print(f"Notice: Checkpoint {checkpoint_path} not found. Running initialized policy.")

    res = evaluate_policy(env, trainer, "ShellMean-MAPPO", seeds=seeds)
    m = res["means"]
    s = res["stds"]

    print("\n--- Evaluation Results Across Seeds ---")
    if len(seeds) > 1:
        print(f"Delivered Traffic:  {m['total_served_mb']:.1f} ± {s['total_served_mb']:.1f} Mb")
        print(f"Throughput:         {m['mean_throughput_mbps']:.2f} ± {s['mean_throughput_mbps']:.2f} Mbps")
        print(f"Episode Reward:     {m['cumulative_reward']:.1f} ± {s['cumulative_reward']:.1f}")
        print(f"Retention Rate:     {m['mean_retention_rate']:.1f}% ± {s['mean_retention_rate']:.1f}%")
        print(f"Conflict Rate:      {m['mean_conflict_rate']:.1f}% ± {s['mean_conflict_rate']:.1f}%")
        print(f"Jain's Fairness:    {m['mean_jains_fairness']:.3f} ± {s['mean_jains_fairness']:.3f}")
        print(f"Interference:       {(m['mean_interference_watts']*1e12):.2f} ± {(s['mean_interference_watts']*1e12):.2f} pW")
    else:
        print(f"Delivered Traffic:  {m['total_served_mb']:.1f} Mb")
        print(f"Throughput:         {m['mean_throughput_mbps']:.2f} Mbps")
        print(f"Episode Reward:     {m['cumulative_reward']:.1f}")
        print(f"Retention Rate:     {m['mean_retention_rate']:.1f}%")
        print(f"Conflict Rate:      {m['mean_conflict_rate']:.1f}%")
        print(f"Jain's Fairness:    {m['mean_jains_fairness']:.3f}")
        print(f"Interference:       {(m['mean_interference_watts']*1e12):.2f} pW")


def mode_compare(config: dict, checkpoint_path: str, seeds: list[int]) -> None:
    """Run baseline comparison experiment and generate comparison plots."""
    print("\n=== Mode: Baseline Comparison ===")
    df_raw = run_baseline_comparison(config, checkpoint_path=checkpoint_path, seeds=seeds)
    plot_baseline_comparison(df_raw, output_path="results/baseline_comparison.png")


def mode_ablation(config: dict, checkpoint_path: str, seeds: list[int]) -> None:
    """Run ablation study and generate ablation plot."""
    print("\n=== Mode: Ablation Study ===")
    df_raw = run_ablation_study(config, checkpoint_path=checkpoint_path, seeds=seeds)
    plot_ablation_comparison(df_raw, output_path="results/ablation_study.png")


def main():
    parser = argparse.ArgumentParser(
        description="ShellMean-MAPPO: Conflict-Aware MARL Framework for Downlink Resource Allocation in Multi-Shell LEO Networks"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="simulate",
        choices=["simulate", "demo", "train", "evaluate", "compare", "ablation", "all"],
        help="Execution mode (simulate/demo, train, evaluate, compare, ablation, all)",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes for simulation or training")
    parser.add_argument("--slots", type=int, default=None, help="Total scheduling slots for training")
    parser.add_argument("--policy", type=str, default="rule_based", choices=["rule_based", "trained", "greedy", "random"], help="Policy for simulate mode")
    parser.add_argument("--eval_interval", type=int, default=128, help="Slots between evaluations during training")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Evaluation random seeds")

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    seeds = args.seeds if args.seeds is not None else config.get("experiment", {}).get("eval_seeds", [42, 101, 202, 303, 404])

    if args.mode in ["simulate", "demo"]:
        num_episodes = args.episodes if args.episodes is not None else 1
        mode_simulate(
            config,
            episodes=num_episodes,
            seed=seeds[0],
            policy_name=args.policy,
            checkpoint_path=args.checkpoint,
        )
    elif args.mode == "train":
        # If --slots is given, use it; otherwise compute from --episodes, or default to 500
        if args.slots is not None:
            training_slots = args.slots
        elif args.episodes is not None:
            ep_len = config.get("network", {}).get("episode_length", 100)
            training_slots = args.episodes * ep_len
        else:
            training_slots = 500
        mode_train(config, slots=training_slots, checkpoint_path=args.checkpoint, eval_interval=args.eval_interval)
    elif args.mode == "evaluate":
        mode_evaluate(config, checkpoint_path=args.checkpoint, seeds=seeds)
    elif args.mode == "compare":
        mode_compare(config, checkpoint_path=args.checkpoint, seeds=seeds)
    elif args.mode == "ablation":
        mode_ablation(config, checkpoint_path=args.checkpoint, seeds=seeds)
    elif args.mode == "all":
        print("\n=== Running Complete End-to-End Experimental Suite ===")
        slots = args.slots if args.slots is not None else 500
        mode_train(config, slots=slots, checkpoint_path=args.checkpoint, eval_interval=args.eval_interval)
        mode_evaluate(config, checkpoint_path=args.checkpoint, seeds=seeds)
        mode_compare(config, checkpoint_path=args.checkpoint, seeds=seeds)
        mode_ablation(config, checkpoint_path=args.checkpoint, seeds=seeds)


if __name__ == "__main__":
    main()
