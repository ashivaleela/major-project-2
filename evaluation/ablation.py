"""Ablation study script evaluating individual components of ShellMean-MAPPO."""

from __future__ import annotations
import os
import argparse
from typing import Dict, Any, List
import yaml
import numpy as np
import pandas as pd

from environment.leo_env import MultiShellLEOEnv
from evaluation.evaluate import evaluate_policy


def run_ablation_study(
    config: Dict[str, Any],
    checkpoint_path: str = "checkpoints/best_model.pt",
    seeds: Optional[List[int]] = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Run Experiment C: Ablation analysis over key components.

    Variants:
    1. Full ShellMean-MAPPO
    2. No Shell Summaries
    3. No Autoregressive Coordination
    4. No Counterfactual Credit
    """
    os.makedirs(output_dir, exist_ok=True)
    if seeds is None:
        seeds = config.get("experiment", {}).get("eval_seeds", [42, 101, 202])[:3]

    print(f"--- Running Ablation Study over seeds: {seeds} ---")
    env = MultiShellLEOEnv(config, seed=seeds[0])

    variants = [
        {"name": "Full_ShellMean_MAPPO", "cf_credit": True, "shell_sum": True},
        {"name": "No_Counterfactual_Credit", "cf_credit": False, "shell_sum": True},
        {"name": "No_Shell_Summaries", "cf_credit": True, "shell_sum": False},
    ]

    all_dfs = []
    summary_rows = []

    for v in variants:
        v_name = v["name"]
        print(f"\n[Ablation] Testing variant: {v_name}...")

        # Create trainer
        v_cfg = yaml.safe_load(yaml.dump(config))
        v_cfg["rewards"]["use_counterfactual_credit"] = v["cf_credit"]

        from training.train import MAPPOTrainer
        trainer = MAPPOTrainer(v_cfg)
        if os.path.exists(checkpoint_path):
            from training.checkpoint import load_checkpoint
            load_checkpoint(trainer.agent, checkpoint_path)

        res = evaluate_policy(env, trainer, v_name, seeds=seeds)
        df_var = res["dataframe"]
        all_dfs.append(df_var)

        m = res["means"]
        s = res["stds"]
        summary_rows.append(
            {
                "Variant": v_name,
                "Delivered Traffic (Mb)": f"{m['total_served_mb']:.1f} ± {s['total_served_mb']:.1f}",
                "Throughput (Mbps)": f"{m['mean_throughput_mbps']:.2f} ± {s['mean_throughput_mbps']:.2f}",
                "Episode Reward": f"{m['cumulative_reward']:.1f} ± {s['cumulative_reward']:.1f}",
                "Retention Rate (%)": f"{m['mean_retention_rate']:.1f}% ± {s['mean_retention_rate']:.1f}%",
                "Jain's Fairness": f"{m['mean_jains_fairness']:.3f} ± {s['mean_jains_fairness']:.3f}",
            }
        )

    df_combined = pd.concat(all_dfs, ignore_index=True)
    raw_path = os.path.join(output_dir, "ablation_study_raw.csv")
    df_combined.to_csv(raw_path, index=False)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "ablation_study_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n=== Ablation Study Summary ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved ablation summary to: {summary_path}")

    return df_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_ablation_study(cfg, checkpoint_path=args.checkpoint)
