# ShellMean-MAPPO: Execution & Demonstration Guide

## Review 2 Demonstration Manual

**Project Title:** ShellMean-MAPPO: A Conflict-Aware Multi-Agent Reinforcement Learning Framework for Downlink Resource Allocation in Multi-Shell LEO Satellite Networks  
**Target Milestone:** Major Project Review 2  
**Operating System:** Windows 10 / 11 (PowerShell)  
**Workspace Root:** `c:\Users\moham\OneDrive\Desktop\projects\Major-Project`  

---

## 1. Project Introduction

### 1.1 Project Title
**ShellMean-MAPPO: A Conflict-Aware Multi-Agent Reinforcement Learning Framework for Downlink Resource Allocation in Multi-Shell LEO Satellite Networks**

### 1.2 Description
Dense Low Earth Orbit (LEO) satellite constellations deployed across multiple orbital shells (e.g., Shell 0 at 550 km, Shell 1 at 1100 km) frequently have overlapping coverage footprints over the same terrestrial ground cells. When multiple satellites independently request downlink transmission to the same ground cell, severe resource contention, co-channel interference, and packet collisions occur.

**ShellMean-MAPPO** resolves this challenge through:
- A **multi-shell constellation simulator** with Keplerian orbital dynamics, dynamic Poisson/burst traffic generation, and finite cell queue buffers.
- An **autoregressive action hierarchy** where satellite agents sequentially select target cells (masked by line-of-sight visibility), non-overlapping Resource Block (RB) chunks (masked by intra-satellite exclusivity), and continuous transmission power shares.
- A **deterministic conflict resolution engine** applying backlog-dependent admission limits ($M_c(t)$), multi-attribute scoring, and deterministic tie-breaking.
- A **physical transmission and co-channel interference model** computing path loss, 3-dB beam pattern roll-off, cross-satellite interference, receiver SINR, and Shannon data rates capped at pending queue backlog.
- A **Centralized Training with Decentralized Execution (CTDE)** architecture using MAPPO, compact orbital-shell summaries, and **Counterfactual Marginal Beam Credit** to resolve multi-agent credit assignment.

### 1.3 Main Objective
To intelligently coordinate satellite downlink beams, spectrum resource blocks, and transmission power to maximize delivered traffic while strictly controlling co-channel interference, contention collisions, buffer backlogs, and service unfairness.

---

## 2. System Requirements

| Parameter | Verified Specification |
|---|---|
| **Operating System** | Windows 10 / 11 64-bit (PowerShell 5.1 / 7+) |
| **Python Version** | Python 3.12.5 (64-bit AMD64) |
| **Deep Learning Framework** | PyTorch (`torch >= 2.0.0`, verified: `2.8.0`) |
| **Numerical Libraries** | `numpy >= 1.22.0` (verified: `2.4.4`), `scipy >= 1.10.0` (verified: `1.16.1`), `pandas >= 1.5.0` (verified: `2.3.2`) |
| **Plotting & Formatting** | `matplotlib >= 3.6.0` (verified: `3.10.8`), `pyyaml >= 6.0` (verified: `6.0.2`) |
| **Environment & Testing** | `gymnasium >= 0.29.0` (verified: `1.4.0`), `pytest >= 7.0.0` (verified: `9.2.0`) |
| **Interactive Dashboard** | `streamlit >= 1.30.0` (verified: `1.49.1`) |
| **Hardware** | Standard x86_64 CPU (4+ cores recommended, 8 GB RAM). GPU is optional; CPU execution is fully optimized. |

---

## 3. Project Folder Structure

```
Major-Project/
├── configs/
│   ├── default.yaml            # Master configuration (network, orbits, traffic, rewards, MAPPO)
│   └── experiment.yaml         # Scalability, ablation, and scenario parameters
├── environment/
│   ├── __init__.py             # Subsystem exports
│   ├── traffic_model.py        # Terrestrial cells, Poisson/burst arrivals, Q(t+1) = max(0, Q - x) + A
│   ├── channel_model.py        # FSPL, atmospheric attenuation, off-axis beam gain, thermal noise
│   ├── satellite_network.py    # Multi-shell constellation, Keplerian circular orbits, visibility
│   ├── resource_manager.py     # Beam feasibility, intra-satellite RB exclusivity, power conservation
│   ├── conflict_resolver.py    # Backlog-dependent admission M_c(t), scoring S(i,k,c), tie-breaking
│   ├── transmission_model.py   # Signal power, cross-satellite co-channel interference, SINR, rate
│   └── leo_env.py              # Gymnasium-compatible multi-agent environment, shell summaries, rewards
├── agents/
│   ├── __init__.py             # Agent exports
│   ├── action_masks.py         # Visibility and intra-satellite RB exclusivity mask manager
│   ├── observation_encoder.py  # Local, ground cell, and orbital shell feature vectors
│   ├── actor.py                # AutoregressiveActor: Cell -> RB Chunk -> Beta Power Share
│   ├── critic.py               # CentralizedCritic estimating V(S) from global network state
│   └── mappo.py                # ShellMeanMAPPO policy wrapper, clipped loss, gradient updates
├── training/
│   ├── __init__.py             # Training exports
│   ├── buffer.py               # RolloutBuffer storing multi-agent transitions, masks, and credits
│   ├── advantages.py           # Generalized Advantage Estimation (GAE) and advantage normalization
│   ├── checkpoint.py           # Safe PyTorch checkpoint saving and loading
│   ├── rollout.py              # Trajectory collector with counterfactual marginal beam credit replay
│   └── train.py                # MAPPOTrainer orchestrating PPO epochs and periodic evaluations
├── baselines/
│   ├── __init__.py             # Baseline exports
│   ├── random_policy.py        # Random feasible scheduling baseline
│   ├── greedy_policy.py        # Greedy backlog/channel gain baseline
│   └── rule_based_policy.py    # Rule-based scheduler with inter-shell RB orthogonalization
├── evaluation/
│   ├── __init__.py             # Evaluation exports
│   ├── metrics.py              # Telemetry tracker (throughput, Jain's fairness, retention, interference)
│   ├── evaluate.py             # Multi-seed episode execution engine
│   ├── compare_baselines.py    # 4-algorithm benchmark comparison suite
│   ├── ablation.py             # Component ablation analysis (shell summaries, counterfactual credit)
│   └── plots.py                # Publication-quality Matplotlib figures
├── tests/
│   ├── test_actor.py           # Autoregressive actor and critic gradient tests
│   ├── test_conflict_resolver.py # Conflict resolution accounting, ranking, and limits
│   ├── test_environment.py     # Reset, step, and non-mutating counterfactual replay tests
│   ├── test_evaluation.py      # Baseline execution and multi-seed evaluation tests
│   ├── test_mappo.py           # GAE calculation, PPO updates, and checkpoint loading
│   ├── test_observations.py    # Feature dimensions and NaN sanitation tests
│   ├── test_resources.py       # Action masks, intra-satellite RB exclusivity, power budget
│   ├── test_traffic.py         # Queue updates, arrival distributions, and backlog bounds
│   └── test_transmission.py    # Desired signal, interference, SINR, and queue-capped delivery
├── checkpoints/
│   ├── best_model.pt           # Verified model weights (Epoch 4, eval_reward: 6154.02)
│   ├── best_model_backup.pt    # Preserved original backup of best model
│   └── latest_model_backup.pt  # Preserved backup copy
├── results/                    # Auto-generated CSV summaries and PNG figures
├── logs/                       # Auto-generated simulation execution and smoke test logs
├── docs/
│   ├── IMPLEMENTATION_STATUS_AUDIT.md       # Complete 11-phase audit
│   ├── REVIEW_2_PROJECT_DOCUMENTATION.md    # Comprehensive 21-section faculty report
│   └── REVIEW_2_VERIFICATION_REPORT.md       # Test execution and verification results
├── app.py                      # Interactive Streamlit dashboard
├── main.py                     # Unified Command-Line Interface
├── README.md                   # Project overview and architecture documentation
├── README_EXECUTION.md         # This execution manual
└── requirements.txt            # Python dependencies
```

---

## 4. Environment Setup (Windows PowerShell)

Open Windows PowerShell and navigate to the project directory:

```powershell
# 1. Navigate to the project root
cd "c:\Users\moham\OneDrive\Desktop\projects\Major-Project"

# 2. Verify current location
Get-Location
```

### 4.1 Creating and Activating Virtual Environment (Optional / Recommended)

If you wish to create a dedicated local virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

> **Note on PowerShell Script Restrictions:**  
> If PowerShell reports `running scripts is disabled on this system`, run this process-scoped bypass (does not affect system security):
> ```powershell
> powershell -ExecutionPolicy Bypass -NoExit -Command ".\venv\Scripts\Activate.ps1"
> ```

### 4.2 Installing Dependencies

Install or verify all required packages:

```powershell
python -m pip install -r requirements.txt
```

Verify that all packages import cleanly:

```powershell
python -c "import torch, yaml, pytest, gymnasium, streamlit, numpy, pandas, matplotlib, scipy; print('SUCCESS: All libraries verified! PyTorch version:', torch.__version__)"
```

### 4.3 Alternative Setup for Linux / macOS

```bash
cd /path/to/Major-Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Execution Instructions

The project provides a unified entry point via `main.py`.

### 5.1 Run the Downlink Simulation Demonstration
Demonstrates multi-satellite beam scheduling, deterministic conflict resolution, physical downlink transmission link budgets, and dynamic traffic queue updates.

```powershell
# Run 1 episode demonstration with default Rule-Based Policy
python main.py --mode simulate --episodes 1

# Run 2 episodes demonstration
python main.py --mode simulate --episodes 2

# Run simulation using the Trained ShellMean-MAPPO Neural Policy
python main.py --mode simulate --policy trained --episodes 1
```

### 5.2 Evaluate the Trained Model Checkpoint
Evaluates `checkpoints/best_model.pt` across multiple evaluation seeds:

```powershell
# Evaluate across default 5 evaluation seeds [42, 101, 202, 303, 404]
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt

# Evaluate over a specific seed
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt --seeds 42
```

### 5.3 Run Benchmark Baseline Comparison
Executes and compares all 4 scheduling algorithms (Random, Greedy, Rule-Based, ShellMean-MAPPO) across evaluation seeds, generates summary tables, and exports `results/baseline_comparison.png`.

```powershell
# Quick comparison over 2 seeds
python main.py --mode compare --seeds 42 101

# Full comparison over 5 seeds
python main.py --mode compare
```

### 5.4 Run Component Ablation Analysis
Evaluates the marginal contribution of key architectural components (Shell Summaries and Counterfactual Credit):

```powershell
python main.py --mode ablation --seeds 42 101
```

### 5.5 Run Training Run
Executes a short PPO training loop with counterfactual credit and GAE advantage normalization:

```powershell
# Short test training run (128 slots)
python main.py --mode train --slots 128 --eval_interval 128

# Multi-episode training run (e.g., 5 episodes = 500 slots)
python main.py --mode train --episodes 5
```

### 5.6 Run Automated Test Suite
Executes all 30 unit and integration tests:

```powershell
python -m pytest tests/ -v
```

### 5.7 Launch Interactive Streamlit Dashboard
Launches the web-based interactive control panel:

```powershell
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 6. Review 2 Demonstration Procedure

Follow this recommended sequence of 4 steps during your Review 2 faculty demonstration:

### Step 1: Run the Automated Test Suite (Code Quality & Verification)
```powershell
python -m pytest tests/ -v
```
- **What happens:** Pytest runs 30 unit and integration tests across actor, critic, conflict resolver, environment, resources, traffic, and transmission models.
- **What to show faculty:** Point out that **30 of 30 tests pass (100%)**, demonstrating mathematical correctness, power conservation, intra-satellite RB exclusivity, and reproducible conflict resolution.

### Step 2: Run the Main Simulation Demonstration (Physical Layer & Conflict Resolution)
```powershell
python main.py --mode simulate --episodes 1
```
- **What happens:** The simulator initializes the dual-shell constellation, generates mid-latitude ground traffic, evaluates visible cells, resolves multi-satellite contention, computes SINR/interference, and updates buffers.
- **What to show faculty:**
  1. The clean summary block titled `========== SHELLMEAN-MAPPO SIMULATION ==========`.
  2. Point out **Conflict Resolution**: How competing requests are scored, retained up to $M_c(t)$, and rejected excess requests produce zero transmission and zero interference.
  3. Show the **Physical Layer Telemetry**: Slant ranges, transmit powers $P_{\text{tx}}$, cross-satellite co-channel interference ($I$), SINR (dB), Shannon rates (Mbps), and delivered traffic capped at buffer backlog.
  4. Mention that the full execution log is permanently recorded in `logs/simulation_execution.log`.

### Step 3: Demonstrate the Trained Neural Policy Inference
```powershell
python main.py --mode simulate --policy trained --episodes 1
```
- **What happens:** Loads `checkpoints/best_model.pt` and demonstrates the trained `AutoregressiveActor` making real-time scheduling decisions.
- **What to show faculty:** The neural actor dynamically selects target cells, allocates RB chunks, and sets continuous power shares via Beta distributions.

### Step 4: Show Baseline Comparison and Generated Figures
```powershell
python main.py --mode compare --seeds 42 101
```
- **What happens:** Runs all 4 policies (Random, Greedy, Rule-Based, ShellMean-MAPPO) and outputs the comparative summary table.
- **What to show faculty:** Open `results/baseline_comparison.png` to show the 4-panel visual comparison across Delivered Traffic, Fairness, Retention Rate, and Backlog.

---

## 7. Checkpoint Information

Three model checkpoint files are maintained in `checkpoints/`:

1. **`checkpoints/best_model.pt`:** The primary model checkpoint loaded during evaluation and simulation. Contains:
   - `epoch`: 4
   - `actor_state_dict`: Autoregressive actor weights (trunk, embeddings, cell/rb/power heads).
   - `critic_state_dict`: Centralized critic weights ($V(S)$ network).
   - `actor_optimizer_state_dict` & `critic_optimizer_state_dict`: Adam optimizer states.
   - `metrics`: Evaluation metrics (`eval_reward`: 6154.02, `eval_throughput`: 68.04 Mbps).
   - `config`: Complete training hyperparameters dictionary.
2. **`checkpoints/best_model_backup.pt`:** An exact safety backup of the original best model weights to guarantee against accidental deletion or overwriting.
3. **`checkpoints/latest_model_backup.pt`:** Additional preserved backup copy.

> **Protection Mechanism:** `training/train.py` has been safeguarded to read the prior checkpoint's best evaluation reward upon initialization. It will **never** overwrite `best_model.pt` unless a new training run achieves an evaluation reward strictly higher than the recorded benchmark (6154.02).

---

## 8. Troubleshooting Guide

| Issue / Error | Root Cause | Solution |
|---|---|---|
| `'pytest' is not recognized as a cmdlet` | The Python user Scripts directory is not in system PATH. | Execute with the module flag: `python -m pytest tests/ -v`. |
| `'streamlit' is not recognized as a cmdlet` | The Streamlit executable is not in system PATH. | Execute with the module flag: `python -m streamlit run app.py`. |
| `ModuleNotFoundError: No module named 'gymnasium'` | Package not installed in the active interpreter. | Run `python -m pip install gymnasium pytest`. |
| `C:\Users\moham\.matplotlib is not a writable directory` | Windows permissions on user folder. | Automatically handled by `main.py` via `MPLCONFIGDIR`. Can also be manually set in PowerShell: `$env:MPLCONFIGDIR = "$env:TEMP\matplotlib"`. |
| Unicode `±` prints as garbled characters | Windows console encoding defaults to cp1252. | Handled automatically in `main.py` via `sys.stdout.reconfigure(encoding="utf-8")`. |
| Checkpoint file not found | Working directory is not project root. | Always run `cd "c:\Users\moham\OneDrive\Desktop\projects\Major-Project"` before executing. |
| Port 8501 already in use (Streamlit) | A previously launched Streamlit process is running. | Run on another port: `python -m streamlit run app.py --server.port 8502`. |

---

## 9. Current Implementation Limitations (Honest Assessment)

1. **Checkpoint Training Stage:** The existing checkpoint (`best_model.pt`) was trained for 4 epochs (512 scheduling slots). The network architecture, forward pass, action masking, and loss updates work 100%, but the policy is still in an exploratory learning phase (delivering ~6004 Mb vs ~14075 Mb for the Rule-Based heuristic). Full policy convergence requiring 10,000–20,000 slots is scheduled for Review 3.
2. **Constellation Geometry:** The orbital propagator uses Keplerian circular orbits with Earth rotation and line-of-sight elevation checks rather than real-time SGP4 TLE orbital propagators.
3. **Terrestrial Region:** Ground cells are distributed in a localized mid-latitude ground zone (40° N) to realistically concentrate multi-satellite visibility within simulation timeframes.

---

## 10. Quick Command Reference Cheat-Sheet

```powershell
# 1. Run automated test suite (30 passed tests)
python -m pytest tests/ -v

# 2. Run simulation demo (Rule-Based Policy, 1 episode)
python main.py --mode simulate --episodes 1

# 3. Run simulation demo (Trained Neural Policy, 1 episode)
python main.py --mode simulate --policy trained --episodes 1

# 4. Evaluate trained model checkpoint across 5 seeds
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt

# 5. Run baseline comparison across algorithms
python main.py --mode compare --seeds 42 101

# 6. Run component ablation study
python main.py --mode ablation --seeds 42 101

# 7. Short training test (128 slots)
python main.py --mode train --slots 128 --eval_interval 128

# 8. Launch interactive Streamlit dashboard
python -m streamlit run app.py
```

