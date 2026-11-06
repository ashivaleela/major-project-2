# ShellMean-MAPPO: Conflict-Aware MARL for Downlink Resource Allocation in Multi-Shell LEO Satellite Networks

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 30 Passed](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)](tests/)
[![Milestone: Review 2 Ready](https://img.shields.io/badge/Review%202-Ready-success.svg)](docs/REVIEW_2_VERIFICATION_REPORT.md)

A comprehensive, modular, and scientifically grounded simulation and multi-agent reinforcement learning (MARL) framework for downlink resource allocation across multi-shell Low Earth Orbit (LEO) satellite constellations.

---

## 🌟 Key Features

1. **Multi-Shell Constellation Dynamics:**
   - Heterogeneous orbital shells (Shell 0 at 550 km, 53° inclination; Shell 1 at 1100 km, 70° inclination) with distinct velocities, path losses, and coverage footprints.
   - Keplerian circular kinematics with Earth rotation and elevation-angle line-of-sight visibility masks.
2. **Deterministic Conflict Resolution (`environment/conflict_resolver.py`):**
   - Dynamic backlog-dependent admission threshold: $M_c(t) = \mathbf{1}[Q_c(t) > 0] \cdot \min(M_{c,\max}, \lceil Q_c(t) / Q_{\text{seg}} \rceil)$.
   - Multi-attribute resolution score combining normalized channel gain, committed power, queue urgency, and shell delay penalty.
   - Deterministic tie-breaking ensuring 100% reproducible request retention and rejection.
3. **Physical RF Transmission & Interference (`environment/transmission_model.py`):**
   - Ka-band (20 GHz) downlink with free-space path loss (FSPL), atmospheric attenuation, and 3-dB beam pattern roll-off.
   - Co-channel interference modeling across satellites sharing overlapping Resource Blocks (RBs).
   - Shannon capacity with implementation efficiency, strictly capped at pending cell buffer queues.
   - Rejected beam requests produce zero transmission and zero interference.
4. **Autoregressive Coordinated Actor & Centralized Critic (`agents/`):**
   - Sequential decision-making: Ground Cell Selection $\to$ RB Chunk Allocation $\to$ Continuous Power Share (Beta distribution).
   - Dynamic action masks enforcing cell visibility and intra-satellite RB exclusivity.
   - Centralized value critic receiving global network state.
5. **Counterfactual Marginal Beam Credit:**
   - Non-mutating replay mechanism evaluating system utility with individual beams deactivated ($U(\mathbf{a}) - U(\mathbf{a}_{-i,k})$), solving the multi-agent multi-beam credit assignment dilemma.
6. **Baselines & Benchmarking (`baselines/`, `evaluation/`):**
   - Executable baseline implementations: Random, Greedy, and Rule-Based Conflict-Aware schedulers.
   - Multi-seed metric aggregations (Throughput, Delivered Data, Retention Rate, Jain's Fairness, Interference, Runtime).
7. **Interactive Dashboard (`app.py`):**
   - Streamlit interface for live slot execution, training progression, baseline comparisons, and ablation visualizer.

---

## 📂 Project Structure

```
Major-Project/
├── configs/
│   ├── default.yaml            # Default network, physical, reward, and MAPPO parameters
│   └── experiment.yaml         # Multi-scenario scalability and ablation configurations
├── environment/
│   ├── traffic_model.py        # Terrestrial cells and dynamic queue updates: Q(t+1) = max(0, Q-x) + A
│   ├── channel_model.py        # Slant range, elevation, FSPL, atmospheric attenuation, antenna gains
│   ├── satellite_network.py    # Multi-shell orbital kinematics, ECEF positions, and visibility
│   ├── resource_manager.py     # Beam, RB, and power constraint validation and intra-satellite exclusivity
│   ├── conflict_resolver.py    # Standalone deterministic multi-satellite conflict resolution
│   ├── transmission_model.py   # Desired signal, co-channel interference, SINR, rate, and queue capping
│   └── leo_env.py              # Multi-agent Gymnasium-compatible environment
├── agents/
│   ├── action_masks.py         # Dynamic masks for cell visibility and RB exclusivity
│   ├── observation_encoder.py  # Local features, ground cells, and ShellMean summary embedding
│   ├── actor.py                # Autoregressive multi-head actor network
│   ├── critic.py               # Centralized critic network estimating V(S)
│   └── mappo.py                # ShellMean-MAPPO agent with clipped PPO objective and credit weights
├── training/
│   ├── buffer.py               # Multi-agent rollout buffer
│   ├── advantages.py           # Generalized Advantage Estimation (GAE) and advantage normalization
│   ├── checkpoint.py           # Checkpoint saving and loading
│   ├── rollout.py              # Trajectory collection with counterfactual beam replay
│   └── train.py                # MAPPOTrainer orchestrating PPO epochs and periodic evaluation
├── baselines/
│   ├── random_policy.py        # Random feasible baseline
│   ├── greedy_policy.py        # Greedy backlog/channel scoring baseline
│   └── rule_based_policy.py    # Rule-based scheduler with inter-shell RB orthogonalization
├── evaluation/
│   ├── metrics.py              # Telemetry tracking (Throughput, Jain's fairness, Retention rate)
│   ├── evaluate.py             # Single and multi-seed episode execution pipeline
│   ├── compare_baselines.py    # Experiment A: Benchmark comparison across all policies
│   ├── ablation.py             # Experiment C: Component ablation study
│   └── plots.py                # Publication-quality figure generation
├── tests/                      # Automated pytest unit and integration test suite (30 passed tests)
├── results/                    # Generated CSV tables and PNG figures
├── checkpoints/                # Saved model weights (best_model.pt) and backups
├── logs/                       # Simulation execution logs (simulation_execution.log)
├── docs/
│   ├── IMPLEMENTATION_STATUS_AUDIT.md       # Complete 11-phase implementation audit
│   ├── REVIEW_2_PROJECT_DOCUMENTATION.md    # Comprehensive 21-section faculty report
│   └── REVIEW_2_VERIFICATION_REPORT.md       # Verification report and test breakdown
├── app.py                      # Interactive Streamlit dashboard
├── main.py                     # Unified Command-Line Interface
├── README.md                   # This project overview
├── README_EXECUTION.md         # Step-by-step Windows execution manual
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.12 (or 3.10+) 64-bit
- PyTorch 2.0+

### 2. Setup Virtual Environment & Dependencies (PowerShell)
```powershell
cd "c:\Users\moham\OneDrive\Desktop\projects\Major-Project"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
python -m pip install -r requirements.txt
```

---

## 💻 Command-Line Interface (CLI)

### 1. Run Automated Tests
```powershell
python -m pytest tests/ -v
```
*Executes all 30 unit and integration tests across traffic, resource manager, conflict resolver, transmission model, actor, critic, MAPPO, and baselines (100% pass).*

### 2. Run Simulator Demonstration
```powershell
# Rule-Based Policy Demonstration (1 episode)
python main.py --mode simulate --episodes 1

# Rule-Based Policy Demonstration (2 episodes)
python main.py --mode simulate --episodes 2

# Trained Neural Policy Demonstration (1 episode)
python main.py --mode simulate --policy trained --episodes 1
```

### 3. Evaluate Trained Checkpoint
```powershell
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
```

### 4. Run Baseline Comparison (Experiment A)
```powershell
python main.py --mode compare --seeds 42 101
```
*Compares Random, Greedy, Rule-Based, and ShellMean-MAPPO across evaluation seeds. Saves `results/baseline_comparison_summary.csv` and `results/baseline_comparison.png`.*

### 5. Run Component Ablation Study (Experiment C)
```powershell
python main.py --mode ablation --seeds 42 101
```
*Evaluates the contribution of Shell Summaries and Counterfactual Beam Credit. Saves `results/ablation_study_summary.csv` and `results/ablation_study.png`.*

### 6. Train ShellMean-MAPPO
```powershell
# Short test training run (128 slots)
python main.py --mode train --slots 128 --eval_interval 128

# Multi-episode training run (5 episodes = 500 slots)
python main.py --mode train --episodes 5
```

### 7. Launch Interactive Streamlit Dashboard
```powershell
python -m streamlit run app.py
```

---

## 📊 Experimental Results

Evaluated across 5 random seeds (`[42, 101, 202, 303, 404]`):

| Algorithm | Delivered Traffic (Mb) | Throughput (Mbps) | Episode Reward | Final Backlog (Mb) | Retention Rate (%) | Conflict Rate (%) | Jain's Fairness | Interference (pW) | Slot Runtime (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **Random** | $13650.9 \pm 229.1$ | $136.51 \pm 2.29$ | $13436.0 \pm 216.0$ | $548.0 \pm 234.5$ | $87.3\% \pm 0.2\%$ | $12.7\% \pm 0.2\%$ | $0.194 \pm 0.006$ | $97.35 \pm 11.16$ | $0.13 \pm 0.01$ |
| **Greedy** | $14036.2 \pm 216.5$ | $140.36 \pm 2.17$ | $13877.1 \pm 222.3$ | $312.3 \pm 14.0$ | $91.1\% \pm 0.6\%$ | $8.9\% \pm 0.6\%$ | $0.269 \pm 0.000$ | $739.67 \pm 30.88$ | $0.08 \pm 0.00$ |
| **Rule-Based** | **$14075.2 \pm 175.8$** | **$140.75 \pm 1.76$** | **$14006.7 \pm 176.9$** | **$273.3 \pm 26.7$** | **$98.8\% \pm 0.5\%$** | **$1.2\% \pm 0.5\%$** | **$0.400 \pm 0.003$** | **$192.03 \pm 4.38$** | $0.09 \pm 0.00$ |
| **ShellMean-MAPPO** *(Epoch 4)* | $6004.1 \pm 486.6$ | $60.04 \pm 4.87$ | $5332.2 \pm 498.7$ | $2309.4 \pm 120.5$ | $67.7\% \pm 0.7\%$ | $32.3\% \pm 0.7\%$ | $0.198 \pm 0.006$ | $336.25 \pm 60.74$ | $14.84 \pm 0.46$ |

---

## 📖 Mandatory Review 2 Deliverables
- 📘 [**Complete Project Documentation (`docs/PROJECT_DOCUMENTATION.md`)**](docs/PROJECT_DOCUMENTATION.md): Consolidated end-to-end project record from problem statement through implementation, execution, verification, results, and next steps.
- 📘 [**Execution & Demonstration Guide (`README_EXECUTION.md`)**](README_EXECUTION.md): Step-by-step PowerShell guide for running tests, live simulation demo, checkpoint evaluation, and Review 2 sequence.
- 📋 [**Implementation Status Audit (`docs/IMPLEMENTATION_STATUS_AUDIT.md`)**](docs/IMPLEMENTATION_STATUS_AUDIT.md): Comprehensive 11-phase audit table and answers to all technical evaluation questions.
- 🎓 [**Academic Review 2 Documentation (`docs/REVIEW_2_PROJECT_DOCUMENTATION.md`)**](docs/REVIEW_2_PROJECT_DOCUMENTATION.md): 21-section academic report with mathematical formulations, architecture diagrams, and experimental results.
- 🔬 [**Verification Report (`docs/REVIEW_2_VERIFICATION_REPORT.md`)**](docs/REVIEW_2_VERIFICATION_REPORT.md): Complete verification log of all executed commands, test breakdowns, and fixed bugs.
