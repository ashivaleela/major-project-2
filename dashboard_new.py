
"""ShellMean-MAPPO Research Dashboard.

Uses the existing simulator and policies. Forecasts are labelled as
heuristic estimates. Failure injection is not claimed unless supported
by the backend.
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


st.set_page_config(
    page_title="ShellMean-MAPPO | Research Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
[data-testid="stAppViewContainer"] {
    background: #0b1020;
}
[data-testid="stSidebar"] {
    background: #111a2e;
}
.hero {
    padding: 1.5rem;
    border: 1px solid #263957;
    border-radius: 16px;
    background: linear-gradient(120deg,#14233d,#172b43,#20213d);
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
    color: #f4f7ff;
}
.hero p {
    margin-top: .5rem;
    color: #b9c8dc;
}
.kpi {
    padding: 1rem;
    border: 1px solid #293b58;
    border-radius: 12px;
    background: #121d31;
    min-height: 105px;
}
.kpi-label {
    font-size: .82rem;
    color: #a8b8cf;
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f4f7ff;
    margin-top: .2rem;
}
.kpi-note {
    font-size: .75rem;
    color: #8fa4c2;
}
div[data-testid="stMetric"] {
    background: #121d31;
    border: 1px solid #293b58;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def make_policy(choice, config, seed):

    if choice == "Rule-Based Conflict-Aware":
        return RuleBasedPolicy(config, seed=seed)

    if choice == "Greedy":
        return GreedyPolicy(config, seed=seed)

    if choice == "Random":
        return RandomPolicy(config, seed=seed)

    from training.train import MAPPOTrainer
    from training.checkpoint import load_checkpoint

    trainer = MAPPOTrainer(config)

    checkpoint = "checkpoints/best_model.pt"

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    load_checkpoint(trainer.agent, checkpoint)

    return trainer


def run_simulation(config, choice, seed, slots):

    env = MultiShellLEOEnv(config, seed=seed)
    policy = make_policy(choice, config, seed)

    obs, info = env.reset(seed=seed)

    rows = []
    last_info = {}

    for slot in range(slots):

        masks = env.get_action_masks()

        if hasattr(policy, "select_actions"):

            try:
                actions = policy.select_actions(
                    obs, masks, env=env
                )
            except TypeError:
                actions = policy.select_actions(obs, masks)

        else:
            actions = policy.step_evaluation(obs, masks)

        next_obs, rewards, terminated, truncated, step_info = (
            env.step(actions)
        )

        rows.append({
            "Slot": slot + 1,
            "Served Traffic (Mb)": float(
                step_info.get("served_traffic_mb", 0)
            ),
            "Queue Backlog (Mb)": float(
                step_info.get("remaining_backlog_mb", 0)
            ),
            "Retained Beams": int(
                step_info.get("num_retained", 0)
            ),
            "Rejected Requests": int(
                step_info.get("num_rejected", 0)
            ),
            "Jain Fairness": float(
                step_info.get("jains_fairness", 0)
            ),
            "Team Reward": float(
                step_info.get("team_reward", 0)
            ),
            "Interference (pW)": float(
                step_info.get("interference_watts", 0)
            ) * 1e12,
            "Mean SINR (dB)": float(
                step_info.get("mean_sinr_db", 0)
            ),
        })

        last_info = step_info
        obs = next_obs

        if terminated or truncated:
            break

    return pd.DataFrame(rows), last_info


def chart_line(df, x, ys, title, y_label):

    fig, ax = plt.subplots(figsize=(9, 3.2))

    for col in ys:
        if col in df.columns:
            ax.plot(
                df[x],
                df[col],
                marker="o",
                linewidth=2,
                label=col,
            )

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.22)

    if len(ys) > 1:
        ax.legend()

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main():

    config = load_config()

    net = config["network"]

    shells = config.get(
        "orbit", {}
    ).get("shell_altitudes_km", [550, 1100])

    inclinations = config.get(
        "orbit", {}
    ).get("inclination_deg", [53, 70])

    sats = int(net.get("num_satellites", 8))
    shell_count = int(net.get("num_shells", 2))

    per_shell = net.get(
        "satellites_per_shell",
        [sats // max(shell_count, 1)] * shell_count,
    )

    beams_per_sat = int(
        net.get("beams_per_satellite", 2)
    )

    cells = int(net.get("num_ground_cells", 20))
    rbs = int(net.get("num_resource_blocks", 8))

    rb_bw = float(
        net.get("rb_bandwidth_mhz", 25)
    )

    total_bw = rbs * rb_bw

    st.markdown("""
    <div class="hero">
        <h1>🛰️ SHELLMEAN-MAPPO</h1>
        <p>Conflict-Aware Adaptive Multi-Agent Reinforcement
        Learning for Downlink Resource Allocation in
        Multi-Shell LEO Satellite Networks</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:

        st.header("Simulation Control")

        policy_name = st.selectbox(
            "Execution Policy",
            [
                "Rule-Based Conflict-Aware",
                "Greedy",
                "Random",
                "Trained ShellMean-MAPPO",
            ],
        )

        seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=99999,
            value=42,
        )

        slots = st.slider(
            "Simulation Slots",
            5,
            min(100, int(net.get("episode_length", 100))),
            20,
        )

        run = st.button(
            "▶ Run Simulation",
            type="primary",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Configured Network")

        st.write(f"**Satellites:** {sats}")
        st.write(f"**Orbital shells:** {shell_count}")
        st.write(f"**Beams:** {sats * beams_per_sat}")
        st.write(f"**Ground cells:** {cells}")
        st.write(f"**Resource blocks:** {rbs}")
        st.write(f"**Spectrum:** {total_bw:g} MHz")

    if "sm_results" not in st.session_state:
        st.session_state.sm_results = None
        st.session_state.sm_policy = None
        st.session_state.sm_seed = None

    if run:

        try:
            with st.spinner("Running satellite network simulation..."):

                df, last_info = run_simulation(
                    config,
                    policy_name,
                    int(seed),
                    int(slots),
                )

            st.session_state.sm_results = {
                "df": df,
                "last": last_info,
            }

            st.session_state.sm_policy = policy_name
            st.session_state.sm_seed = int(seed)

        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    result = st.session_state.sm_results

    if result is None:

        st.info(
            "Select a policy and click Run Simulation "
            "to populate the research dashboard."
        )

        df = pd.DataFrame()
        last = {}

    else:

        df = result["df"]
        last = result["last"]

        st.caption(
            f"Latest run: {st.session_state.sm_policy} | "
            f"Seed: {st.session_state.sm_seed} | "
            f"Slots: {len(df)}"
        )

    tabs = st.tabs([
        "01 · Network Overview",
        "02 · Live Satellite Network",
        "03 · Resource Allocation",
        "04 · MAPPO Performance",
        "05 · Failure & Adaptation Lab",
        "06 · Traffic Prediction",
    ])

    # --------------------------------------------------
    # 1. NETWORK OVERVIEW
    # --------------------------------------------------

    with tabs[0]:

        st.subheader("Network Overview")

        c = st.columns(4)

        with c[0]:
            metric_card(
                "Satellites",
                str(sats),
                f"{shell_count} orbital shells",
            )

        with c[1]:
            metric_card(
                "Beam Capacity",
                str(sats * beams_per_sat),
                f"{beams_per_sat} beams per satellite",
            )

        with c[2]:
            metric_card(
                "Total Spectrum",
                f"{total_bw:g} MHz",
                f"{rbs} × {rb_bw:g} MHz RBs",
            )

        with c[3]:
            metric_card(
                "Ground Cells",
                str(cells),
                "Terrestrial demand regions",
            )

        st.markdown("### Latest Simulation Performance")

        if not df.empty:

            total = df["Served Traffic (Mb)"].sum()

            throughput = total / max(len(df), 1)

            retention = (
                df["Retained Beams"].sum()
                /
                max(
                    df["Retained Beams"].sum()
                    + df["Rejected Requests"].sum(),
                    1,
                )
            ) * 100

            m = st.columns(4)

            m[0].metric(
                "Delivered Traffic",
                f"{total:,.1f} Mb",
            )

            m[1].metric(
                "Mean Throughput",
                f"{throughput:,.2f} Mbps",
            )

            m[2].metric(
                "Request Retention",
                f"{retention:.1f}%",
            )

            m[3].metric(
                "Mean Jain Fairness",
                f"{df['Jain Fairness'].mean():.3f}",
            )

            chart_line(
                df,
                "Slot",
                ["Served Traffic (Mb)", "Queue Backlog (Mb)"],
                "Service and Queue Evolution",
                "Mb",
            )

        else:

            st.info(
                "Run a simulation to display measured performance."
            )

        st.markdown("### Research Objective")

        st.write(
            "Coordinate downlink resources across multiple LEO "
            "shells while managing satellite visibility, traffic "
            "demand, interference and competing beam requests."
        )

    # --------------------------------------------------
    # 2. LIVE SATELLITE NETWORK
    # --------------------------------------------------

    with tabs[1]:

        st.subheader("Two-Shell Satellite Network")

        st.caption(
            "Conceptual topology based on configured shell counts. "
            "This is not a live orbital propagator."
        )

        fig, ax = plt.subplots(figsize=(10, 4.6))

        ax.set_facecolor("#0b1020")
        ax.axis("off")

        for si in range(shell_count):

            y = 0.72 if si == 0 else 0.42

            count = (
                int(per_shell[si])
                if si < len(per_shell)
                else 0
            )

            xs = np.linspace(
                0.15,
                0.85,
                max(count, 1),
            )

            ax.plot(
                [0.08, 0.92],
                [y, y],
                linestyle="--",
                alpha=0.35,
            )

            for j, xpos in enumerate(xs[:count]):

                ax.scatter(
                    xpos,
                    y,
                    s=260,
                    marker="s",
                    edgecolors="white",
                )

                ax.text(
                    xpos,
                    y + 0.07,
                    f"S{si+1}-{j+1}",
                    ha="center",
                    color="white",
                    fontsize=9,
                )

            altitude = (
                shells[si]
                if si < len(shells)
                else "n/a"
            )

            inclination = (
                inclinations[si]
                if si < len(inclinations)
                else "n/a"
            )

            ax.text(
                0.02,
                y - 0.09,
                f"Shell {si+1}: {altitude} km · {inclination}°",
                color="white",
                fontsize=9,
            )

        ground_x = np.linspace(0.12, 0.88, cells)

        ax.scatter(
            ground_x,
            np.full(cells, 0.12),
            marker="^",
            s=45,
        )

        for i, xpos in enumerate(ground_x):

            ax.text(
                xpos,
                0.045,
                f"C{i+1}",
                ha="center",
                color="white",
                fontsize=6,
            )

        ax.text(
            0.5,
            0.25,
            f"{cells} ground cells · {rbs} RBs · "
            f"{sats * beams_per_sat} beams",
            ha="center",
            color="white",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.warning(
            "Actual satellite-to-cell visibility and geographic "
            "positions are not exposed by the current dashboard "
            "interface. Connection lines are therefore not fabricated."
        )

        st.write({
            "Shell altitudes (km)": shells,
            "Inclinations (degrees)": inclinations,
            "Satellites per shell": per_shell,
        })

    # --------------------------------------------------
    # 3. RESOURCE ALLOCATION
    # --------------------------------------------------

    with tabs[2]:

        st.subheader("Resource Allocation")

        st.write(
            "The existing policy selects downlink actions. "
            "The resource manager and conflict resolver enforce "
            "allocation constraints."
        )

        if not df.empty:

            cols = [
                "Slot",
                "Retained Beams",
                "Rejected Requests",
                "Interference (pW)",
                "Mean SINR (dB)",
            ]

            st.dataframe(
                df[cols],
                use_container_width=True,
                hide_index=True,
            )

            chart_line(
                df,
                "Slot",
                ["Retained Beams", "Rejected Requests"],
                "Beam Admission Decisions",
                "Count",
            )

            chart_line(
                df,
                "Slot",
                ["Interference (pW)", "Mean SINR (dB)"],
                "Radio Conditions",
                "Measured Value",
            )

        else:

            st.info(
                "Run a simulation to populate resource allocation metrics."
            )

        st.markdown("### Configured Resource Limits")

        st.write(
            f"- {rbs} resource blocks × {rb_bw:g} MHz "
            f"= {total_bw:g} MHz total bandwidth"
        )

        st.write(
            f"- {sats * beams_per_sat} total beam capacity"
        )

        st.write(
            "- Resource-block exclusivity and power-budget "
            "enforcement are handled by the existing backend."
        )

    # --------------------------------------------------
    # 4. MAPPO PERFORMANCE
    # --------------------------------------------------

    with tabs[3]:

        st.subheader("MAPPO Training and Policy Performance")

        st.caption(
            "Reports available experiment artifacts and the "
            "latest selected-policy rollout."
        )

        training_csv = "results/training_log.csv"

        benchmark_csv = (
            "results/baseline_comparison_summary.csv"
        )

        left, right = st.columns(2)

        with left:

            if os.path.exists(training_csv):

                train_df = pd.read_csv(training_csv)

                st.write("Training Log")

                st.dataframe(
                    train_df.tail(12),
                    use_container_width=True,
                    hide_index=True,
                )

                numeric = train_df.select_dtypes(
                    include=np.number
                )

                if not numeric.empty:

                    ycols = [
                        c for c in numeric.columns
                        if "reward" in c.lower()
                    ]

                    if ycols:

                        plot_df = numeric.reset_index()

                        plot_df = plot_df.rename(
                            columns={"index": "Iteration"}
                        )

                        chart_line(
                            plot_df,
                            "Iteration",
                            ycols[:3],
                            "Training Reward",
                            "Reward",
                        )

            else:

                st.info(
                    "No training_log.csv found. Run the training "
                    "pipeline to generate convergence history."
                )

        with right:

            if os.path.exists(benchmark_csv):

                bench = pd.read_csv(benchmark_csv)

                st.write("Baseline Comparison")

                st.dataframe(
                    bench,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No baseline comparison CSV found. "
                    "Run the evaluation pipeline to generate results."
                )

        if not df.empty:

            st.markdown("### Current Rollout")

            st.metric(
                "Mean Team Reward",
                f"{df['Team Reward'].mean():.3f}",
            )

            chart_line(
                df,
                "Slot",
                ["Team Reward", "Jain Fairness"],
                "Reward and Fairness per Slot",
                "Value",
            )

        st.warning(
            "The saved checkpoint is an early training checkpoint. "
            "Do not claim MAPPO outperforms Rule-Based unless fresh, "
            "reproducible benchmark results establish that."
        )

    # --------------------------------------------------
    # 5. FAILURE AND ADAPTATION LAB
    # --------------------------------------------------

    with tabs[4]:

        st.subheader("Failure and Adaptation Lab")

        st.write(
            "This section is reserved for controlled satellite "
            "and communication-link failure experiments."
        )

        st.info(
            "Failure injection is not enabled yet. The currently "
            "exposed simulator API does not provide a verified "
            "satellite/link failure injection method."
        )

        st.markdown("### Backend Extension Required")

        st.code(
            "1. Add satellite/link availability state.\n"
            "2. Exclude failed links from visibility and action masks.\n"
            "3. Re-run allocation after failure.\n"
            "4. Measure service recovery.\n"
            "5. Compare normal, failure and recovery scenarios.",
            language="text",
        )

        if not df.empty:

            st.markdown(
                "### Current Run: Service Continuity Indicators"
            )

            chart_line(
                df,
                "Slot",
                ["Served Traffic (Mb)", "Queue Backlog (Mb)"],
                "Observed Service Under Selected Policy",
                "Mb",
            )

            st.caption(
                "These are normal-run observations, not a "
                "failure-injection experiment."
            )

    # --------------------------------------------------
    # 6. TRAFFIC PREDICTION
    # --------------------------------------------------

    with tabs[5]:

        st.subheader("Traffic Demand and Proactive Allocation")

        st.write(
            "The simulator reports served traffic and queue backlog. "
            "The following moving average is a simple heuristic "
            "derived from observed service, not a trained predictor."
        )

        if not df.empty:

            series = df[
                ["Slot", "Served Traffic (Mb)"]
            ].copy()

            series["Observed Service (Mb)"] = (
                series["Served Traffic (Mb)"]
            )

            series["3-Slot Moving Average (Mb)"] = (
                series["Served Traffic (Mb)"]
                .rolling(3, min_periods=1)
                .mean()
            )

            chart_line(
                series,
                "Slot",
                [
                    "Observed Service (Mb)",
                    "3-Slot Moving Average (Mb)",
                ],
                "Observed Service and Smoothed Trend",
                "Mb per Slot",
            )

            latest = float(
                series["3-Slot Moving Average (Mb)"].iloc[-1]
            )

            st.metric(
                "Latest Heuristic Trend",
                f"{latest:.2f} Mb/slot",
            )

            st.caption(
                "The moving average smooths recent served traffic. "
                "It does not predict unseen arrivals or alter "
                "resource-allocation actions."
            )

        else:

            st.info(
                "Run a simulation to calculate the observed traffic trend."
            )

        st.markdown("### Full Abstract-Level Feature Still Required")

        st.write(
            "A genuine proactive MARL extension needs a demand "
            "predictor trained or evaluated on arrival history, "
            "then a verified interface that passes its forecast "
            "into the allocation policy."
        )

    st.divider()

    st.caption(
        "ShellMean-MAPPO · Simulation-based research prototype · "
        "Metrics are shown only when produced by the local backend."
    )


if __name__ == "__main__":
    main()
