"""Benchmark comparison script: Random, Greedy, Rule-Based, and ShellMean-MAPPO."""

from __future__ import annotations
import os
import argparse
from typing import Dict, Any, List
import yaml
import numpy as np
import pandas as pd

from environment.leo_env import MultiShellLEOEnv
from baselines.random_policy import RandomPolicy
from baselines.greedy_policy import GreedyPolicy
from baselines.rule_based_policy import RuleBasedPolicy
from evaluation.evaluate import evaluate_policy


def run_baseline_comparison(
    config: Dict[str, Any],
    checkpoint_path: str = "checkpoints/best_model.pt",
    seeds: Optional[List[int]] = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Run Experiment A: Evaluate and compare all 4 scheduling algorithms.

    Args:
        config: Full configuration dictionary.
        checkpoint_path: Path to trained ShellMean-MAPPO model weights.
        seeds: List of evaluation seeds (default: 5 seeds from config).
        output_dir: Directory to save results.

    Returns:
        DataFrame summarizing comparison results.
    """
    os.makedirs(output_dir, exist_ok=True)
    if seeds is None:
        seeds = config.get("experiment", {}).get("eval_seeds", [42, 101, 202, 303, 404])

    print(f"--- Running Baseline Comparison over {len(seeds)} seeds: {seeds} ---")

    # Initialize environment
    env = MultiShellLEOEnv(config, seed=seeds[0])

    # 1. Random Policy
    print("[1/4] Evaluating Random Baseline...")
    rand_policy = RandomPolicy(config, seed=seeds[0])
    res_rand = evaluate_policy(env, rand_policy, "Random", seeds=seeds)

    # 2. Greedy Policy
    print("[2/4] Evaluating Greedy Baseline...")
    greedy_policy = GreedyPolicy(config, seed=seeds[0])
    res_greedy = evaluate_policy(env, greedy_policy, "Greedy", seeds=seeds)

    # 3. Rule-Based Conflict-Aware Policy
    print("[3/4] Evaluating Rule-Based Policy...")
    rule_policy = RuleBasedPolicy(config, seed=seeds[0])
    res_rule = evaluate_policy(env, rule_policy, "Rule-Based", seeds=seeds)

    # 4. Proposed ShellMean-MAPPO
    print("[4/4] Evaluating ShellMean-MAPPO...")
    from training.train import MAPPOTrainer
    from training.checkpoint import load_checkpoint
    mappo_trainer = MAPPOTrainer(config)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        load_checkpoint(mappo_trainer.agent, checkpoint_path)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using initialized agent.")
    res_mappo = evaluate_policy(env, mappo_trainer, "ShellMean-MAPPO", seeds=seeds)

    # Combine raw records
    df_all = pd.concat(
        [res_rand["dataframe"], res_greedy["dataframe"], res_rule["dataframe"], res_mappo["dataframe"]],
        ignore_index=True,
    )
    raw_path = os.path.join(output_dir, "baseline_comparison_raw.csv")
    df_all.to_csv(raw_path, index=False)
    print(f"Saved raw episode results to: {raw_path}")

    # Build formatted summary table
    summary_rows = []
    for res in [res_rand, res_greedy, res_rule, res_mappo]:
        name = res["policy"]
        m = res["means"]
        s = res["stds"]
        summary_rows.append(
            {
                "Algorithm": name,
                "Delivered Traffic (Mb)": f"{m['total_served_mb']:.1f} ± {s['total_served_mb']:.1f}",
                "Throughput (Mbps)": f"{m['mean_throughput_mbps']:.2f} ± {s['mean_throughput_mbps']:.2f}",
                "Episode Reward": f"{m['cumulative_reward']:.1f} ± {s['cumulative_reward']:.1f}",
                "Final Backlog (Mb)": f"{m['final_backlog_mb']:.1f} ± {s['final_backlog_mb']:.1f}",
                "Retention Rate (%)": f"{m['mean_retention_rate']:.1f}% ± {s['mean_retention_rate']:.1f}%",
                "Conflict Rate (%)": f"{m['mean_conflict_rate']:.1f}% ± {s['mean_conflict_rate']:.1f}%",
                "Jain's Fairness": f"{m['mean_jains_fairness']:.3f} ± {s['mean_jains_fairness']:.3f}",
                "Interference (pW)": f"{(m['mean_interference_watts']*1e12):.2f} ± {(s['mean_interference_watts']*1e12):.2f}",
                "Slot Runtime (ms)": f"{m['mean_runtime_ms']:.2f} ± {s['mean_runtime_ms']:.2f}",
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "baseline_comparison_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print("\n=== Baseline Comparison Summary ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")

    return df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_baseline_comparison(cfg, checkpoint_path=args.checkpoint)
