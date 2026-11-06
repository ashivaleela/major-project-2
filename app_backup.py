"""Interactive Streamlit Dashboard for ShellMean-MAPPO.

Visualizes multi-shell LEO downlink resource allocation, conflict resolution,
physical layer interference, training progression, and baseline benchmarks.
"""

from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from environment.leo_env import MultiShellLEOEnv
from baselines.random_policy import RandomPolicy
from baselines.greedy_policy import GreedyPolicy
from baselines.rule_based_policy import RuleBasedPolicy
from evaluation.evaluate import run_episode


st.set_page_config(
    page_title="ShellMean-MAPPO Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    st.title("🛰️ ShellMean-MAPPO: Satellite Downlink Resource Allocation")
    st.caption(
        "A Conflict-Aware Multi-Agent Reinforcement Learning Framework for Multi-Shell LEO Networks "
        "(Simulation-Based Research Model)"
    )

    config = load_config()

    # Sidebar Controls
    st.sidebar.header("Network Parameters")
    num_sats = config["network"]["num_satellites"]
    num_shells = config["network"]["num_shells"]
    num_cells = config["network"]["num_ground_cells"]
    num_rbs = config["network"]["num_resource_blocks"]

    st.sidebar.info(
        f"**Constellation:** {num_sats} Satellites ({num_shells} Shells)\n\n"
        f"**Beams:** 2 per Satellite ({num_sats * 2} Total)\n\n"
        f"**Ground Cells:** {num_cells} Terrestrial Cells\n\n"
        f"**Spectrum:** {num_rbs} Resource Blocks (200 MHz)"
    )

    policy_choice = st.sidebar.selectbox(
        "Interactive Policy",
        options=["Rule-Based Conflict-Aware", "Greedy Policy", "Random Policy", "Trained ShellMean-MAPPO"],
        index=0,
    )

    sim_seed = st.sidebar.slider("Simulation Seed", 1, 500, 42)
    sim_slots = st.sidebar.slider("Slots to Simulate", 5, 50, 10)

    # Main Tabs
    tab_sim, tab_train, tab_benchmarks, tab_ablation, tab_arch = st.tabs(
        [
            "⚡ Live Slot Execution",
            "📈 Training Convergence",
            "📊 Baseline Benchmarks",
            "🔬 Component Ablation",
            "📐 System Architecture",
        ]
    )

    with tab_sim:
        st.subheader(f"Interactive Execution ({policy_choice})")

        env = MultiShellLEOEnv(config, seed=sim_seed)

        # Select policy
        if policy_choice == "Random Policy":
            policy = RandomPolicy(config, seed=sim_seed)
        elif policy_choice == "Greedy Policy":
            policy = GreedyPolicy(config, seed=sim_seed)
        elif policy_choice == "Rule-Based Conflict-Aware":
            policy = RuleBasedPolicy(config, seed=sim_seed)
        else:
            from training.train import MAPPOTrainer
            from training.checkpoint import load_checkpoint

            trainer = MAPPOTrainer(config)
            chk_path = "checkpoints/best_model.pt"
            if os.path.exists(chk_path):
                load_checkpoint(trainer.agent, chk_path)
            policy = trainer

        # Run slots
        with st.spinner("Simulating downlink slots..."):
            obs, info = env.reset(seed=sim_seed)
            history_slots = []
            for slot_idx in range(sim_slots):
                masks = env.get_action_masks()
                if hasattr(policy, "select_actions"):
                    try:
                        actions = policy.select_actions(obs, masks, env=env)
                    except TypeError:
                        actions = policy.select_actions(obs, masks)
                else:
                    actions = policy.step_evaluation(obs, masks)

                next_obs, rewards, term, trunc, step_info = env.step(actions)
                history_slots.append(
                    {
                        "Slot": slot_idx + 1,
                        "Served Traffic (Mb)": step_info["served_traffic_mb"],
                        "Queue Backlog (Mb)": step_info["remaining_backlog_mb"],
                        "Retained Beams": step_info["num_retained"],
                        "Rejected Requests": step_info["num_rejected"],
                        "Jain's Fairness": step_info["jains_fairness"],
                        "Team Reward": step_info["team_reward"],
                        "Interference (pW)": step_info["interference_watts"] * 1e12,
                        "Mean SINR (dB)": step_info["mean_sinr_db"],
                    }
                )
                obs = next_obs
                if term or trunc:
                    break

            df_sim = pd.DataFrame(history_slots)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Delivered", f"{df_sim['Served Traffic (Mb)'].sum():.1f} Mb")
        col2.metric("Mean Throughput", f"{df_sim['Served Traffic (Mb)'].mean():.2f} Mbps")
        col3.metric("Avg Retention Rate", f"{(df_sim['Retained Beams'].sum() / max(1, df_sim['Retained Beams'].sum() + df_sim['Rejected Requests'].sum()) * 100):.1f}%")
        col4.metric("Avg Fairness", f"{df_sim['Jain\'s Fairness'].mean():.3f}")

        st.dataframe(df_sim, use_container_width=True)

        # Plot slot dynamics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(df_sim["Slot"], df_sim["Served Traffic (Mb)"], marker="o", color="#2ca02c", label="Served Traffic (Mb)")
        ax1.plot(df_sim["Slot"], df_sim["Queue Backlog (Mb)"], marker="s", color="#d62728", label="Queue Backlog (Mb)")
        ax1.set_title("Traffic & Queue Backlog Dynamics")
        ax1.set_xlabel("Scheduling Slot")
        ax1.set_ylabel("Megabits (Mb)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(df_sim["Slot"], df_sim["Retained Beams"], marker="^", color="#1f77b4", label="Retained Beams")
        ax2.plot(df_sim["Slot"], df_sim["Rejected Requests"], marker="x", color="#ff7f0e", label="Rejected Requests")
        ax2.set_title("Conflict Resolution Decisions")
        ax2.set_xlabel("Scheduling Slot")
        ax2.set_ylabel("Beam Count")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        st.pyplot(fig)
        plt.close(fig)

    with tab_train:
        st.subheader("Training Convergence Curves")
        log_path = "results/training_log.csv"
        plot_path = "results/training_curves.png"

        if os.path.exists(log_path):
            df_train = pd.read_csv(log_path)
            st.dataframe(df_train.tail(10), use_container_width=True)
            if os.path.exists(plot_path):
                st.image(plot_path, caption="ShellMean-MAPPO Training Progression", use_container_width=True)
        else:
            st.info("No training log found yet. Run `python main.py --mode train` to generate training records.")

    with tab_benchmarks:
        st.subheader("Baseline Performance Comparison")
        summary_path = "results/baseline_comparison_summary.csv"
        plot_bench = "results/baseline_comparison.png"

        if os.path.exists(summary_path):
            df_b = pd.read_csv(summary_path)
            st.table(df_b)
            if os.path.exists(plot_bench):
                st.image(plot_bench, caption="Algorithm Comparison across 5 Evaluation Seeds", use_container_width=True)
        else:
            st.info("No baseline comparison results found yet. Run `python main.py --mode compare`.")

    with tab_ablation:
        st.subheader("Ablation Analysis")
        ablation_summary = "results/ablation_study_summary.csv"
        plot_abl = "results/ablation_study.png"

        if os.path.exists(ablation_summary):
            df_a = pd.read_csv(ablation_summary)
            st.table(df_a)
            if os.path.exists(plot_abl):
                st.image(plot_abl, caption="Ablation Analysis: Shell Summaries & Counterfactual Credit", use_container_width=True)
        else:
            st.info("No ablation study results found yet. Run `python main.py --mode ablation`.")

    with tab_arch:
        st.subheader("System Architecture & Mathematical Formulation")
        st.markdown(
            """
            ### 1. Multi-Shell Constellation Geometry
            - **Shell 0 (Low LEO):** Altitude 550 km, Inclination 53°, lower propagation delay, localized footprint.
            - **Shell 1 (High LEO):** Altitude 1100 km, Inclination 70°, broader footprint, higher path loss.
            
            ### 2. Backlog-Dependent Conflict Resolution
            When multiple satellites request service to the same cell $c$:
            $$M_c(t) = \\mathbf{1}[Q_c(t) > 0] \\cdot \\min\\left(M_{c,\\max}, \\left\\lceil \\frac{Q_c(t)}{Q_{\\text{seg}}} \\right\\rceil\\right)$$
            
            Multi-attribute scoring:
            $$S(i, k, c) = w_h \\frac{h_{i,c}}{\\bar{h}} + w_p p_{i,k} + w_q \\frac{Q_c(t)}{Q_{\\max}} - w_\\tau \\frac{\\tau_i}{\\tau_{\\max}}$$
            Deterministic tie-breaking: `(-score, sat_id, beam_id)`.
            
            ### 3. Autoregressive Action Hierarchy
            1. Target Ground Cell: $a_{i,k}^{\\text{cell}} \\in \\{0, 1, \\dots, N_{\\text{cells}}\\}$ (Masked by visibility).
            2. RB Chunk Allocation: $(s_{i,k}, w_{i,k})$ (Masked by intra-satellite exclusivity).
            3. Transmit Power Share: $p_{i,k} \\in [0, 1]$ subject to $\\sum_k p_{i,k} + p_{i,0} = 1.0$.
            
            ### 4. Counterfactual Marginal Beam Credit
            For active beam $k$ of satellite $i$:
            $$C_{i,k} = U(\\mathbf{a}) - U(\\mathbf{a}_{-i,k})$$
            Replays the slot with beam $(i,k)$ turned off (power shifted to reserved), isolating the beam's true marginal utility contribution!
            """
        )


if __name__ == "__main__":
    main()

