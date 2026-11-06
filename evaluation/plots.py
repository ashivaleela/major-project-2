"""Plot generation for evaluation benchmarks, training curves, and network dynamics."""

from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(df_history: pd.DataFrame, output_path: str = "results/training_curves.png") -> None:
    """Plot multi-panel training progression curves."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("ShellMean-MAPPO Training Convergence", fontsize=14, fontweight="bold")

    slots = df_history["slots_trained"] if "slots_trained" in df_history.columns else df_history.index

    # 1. Rollout Reward
    axes[0, 0].plot(slots, df_history["rollout_reward"], color="#1f77b4", lw=2, label="Rollout Reward")
    if "eval_reward" in df_history.columns and df_history["eval_reward"].notna().any():
        axes[0, 0].plot(slots, df_history["eval_reward"], color="#ff7f0e", lw=2, marker="o", label="Eval Reward")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Scheduling Slots")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend()

    # 2. Served Traffic
    axes[0, 1].plot(slots, df_history["rollout_served_mb"], color="#2ca02c", lw=2)
    axes[0, 1].set_title("Delivered Traffic per Slot (Mb)")
    axes[0, 1].set_xlabel("Scheduling Slots")
    axes[0, 1].set_ylabel("Traffic (Mb)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 3. Request Retention Rate
    axes[1, 0].plot(slots, df_history["rollout_retention"], color="#9467bd", lw=2)
    axes[1, 0].set_title("Request Retention Rate (%)")
    axes[1, 0].set_xlabel("Scheduling Slots")
    axes[1, 0].set_ylabel("Retention (%)")
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # 4. Losses
    axes[1, 1].plot(slots, df_history["actor_loss"], color="#d62728", lw=1.5, label="Actor Loss")
    axes[1, 1].plot(slots, df_history["critic_loss"], color="#8c564b", lw=1.5, label="Critic Loss")
    axes[1, 1].set_title("PPO Loss Progression")
    axes[1, 1].set_xlabel("Scheduling Slots")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves plot to: {output_path}")


def plot_baseline_comparison(
    df_raw: pd.DataFrame,
    output_path: str = "results/baseline_comparison.png",
) -> None:
    """Generate 4-panel comparison bar charts with error bars across algorithms."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Performance Comparison: ShellMean-MAPPO vs Baselines", fontsize=15, fontweight="bold")

    policies = df_raw["policy"].unique()
    colors = ["#7f7f7f", "#d62728", "#1f77b4", "#2ca02c"]

    # 1. Delivered Traffic
    means = df_raw.groupby("policy")["total_served_mb"].mean().reindex(policies)
    stds = df_raw.groupby("policy")["total_served_mb"].std().reindex(policies)
    axes[0, 0].bar(policies, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 0].set_title("Total Delivered Traffic (Mb)")
    axes[0, 0].set_ylabel("Traffic (Mb)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)

    # 2. Jain's Fairness Index
    means = df_raw.groupby("policy")["mean_jains_fairness"].mean().reindex(policies)
    stds = df_raw.groupby("policy")["mean_jains_fairness"].std().reindex(policies)
    axes[0, 1].bar(policies, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 1].set_title("Jain's Fairness Index")
    axes[0, 1].set_ylabel("Fairness [0, 1]")
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.6)

    # 3. Request Retention Rate
    means = df_raw.groupby("policy")["mean_retention_rate"].mean().reindex(policies)
    stds = df_raw.groupby("policy")["mean_retention_rate"].std().reindex(policies)
    axes[1, 0].bar(policies, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title("Request Retention Rate (%)")
    axes[1, 0].set_ylabel("Retention (%)")
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.6)

    # 4. Final Queue Backlog
    means = df_raw.groupby("policy")["final_backlog_mb"].mean().reindex(policies)
    stds = df_raw.groupby("policy")["final_backlog_mb"].std().reindex(policies)
    axes[1, 1].bar(policies, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 1].set_title("Final Queue Backlog (Mb)")
    axes[1, 1].set_ylabel("Backlog (Mb)")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved baseline comparison plot to: {output_path}")


def plot_ablation_comparison(
    df_raw: pd.DataFrame,
    output_path: str = "results/ablation_study.png",
) -> None:
    """Generate ablation study comparison chart."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Ablation Analysis: Component Impact", fontsize=14, fontweight="bold")

    variants = df_raw["policy"].unique()
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]

    # 1. Delivered Traffic
    means = df_raw.groupby("policy")["total_served_mb"].mean().reindex(variants)
    stds = df_raw.groupby("policy")["total_served_mb"].std().reindex(variants)
    axes[0].bar(variants, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Total Delivered Traffic (Mb)")
    axes[0].set_ylabel("Traffic (Mb)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # 2. Cumulative Reward
    means = df_raw.groupby("policy")["cumulative_reward"].mean().reindex(variants)
    stds = df_raw.groupby("policy")["cumulative_reward"].std().reindex(variants)
    axes[1].bar(variants, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title("Cumulative Episode Reward")
    axes[1].set_ylabel("Reward")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    # 3. Retention Rate
    means = df_raw.groupby("policy")["mean_retention_rate"].mean().reindex(variants)
    stds = df_raw.groupby("policy")["mean_retention_rate"].std().reindex(variants)
    axes[2].bar(variants, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[2].set_title("Request Retention Rate (%)")
    axes[2].set_ylabel("Retention (%)")
    axes[2].set_ylim([0, 105])
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved ablation plot to: {output_path}")

