# ShellMean-MAPPO: Complete Project Documentation

## Project title

**ShellMean-MAPPO: A Conflict-Aware Multi-Agent Reinforcement Learning Framework for Downlink Resource Allocation in Multi-Shell Low Earth Orbit Satellite Networks**

## Document purpose and status

This document provides one end-to-end account of the project: the problem being addressed, the proposed system, its software implementation, the recorded engineering work, how to run it, verification evidence, current results, and remaining work. It consolidates information already present in the project documentation and implementation. The benchmark figures below are historical figures recorded in those reports; they were not regenerated for this documentation update.

## 1. Abstract

Low Earth Orbit (LEO) satellite constellations can provide broadband access to areas that are difficult to serve with terrestrial infrastructure. In a multi-shell constellation, satellites at different altitudes may simultaneously see and attempt to serve the same ground cell. With limited beams, spectrum, and transmit power, independent scheduling can result in excessive contention, co-channel interference, poor service fairness, and traffic queues that are not drained.

This project builds a configurable simulation and multi-agent reinforcement learning (MARL) framework for coordinated satellite downlink scheduling. Its main components are a two-shell satellite-network simulator, dynamic ground-cell traffic queues, a physical channel and interference model, deterministic conflict resolution, an autoregressive MAPPO actor with a centralized critic, baseline schedulers, an evaluation pipeline, and a Streamlit dashboard. A counterfactual beam-credit mechanism is included to estimate the marginal utility of individual beam decisions.

The project has working simulation, evaluation, training, and demonstration entry points, as well as an existing trained checkpoint. The recorded verification reports state that all 30 automated tests passed. However, the available checkpoint is an early training result and the recorded benchmark shows that the rule-based baseline performs substantially better than the MAPPO policy at that training stage. The project is therefore an implemented and tested research prototype, not evidence that the learned policy has converged or outperformed the baselines.

## 2. Problem statement

At each scheduling interval, multiple satellites can have line-of-sight visibility to the same ground cell. Each satellite must decide which cell to serve, which radio resource blocks (RBs) to allocate to each beam, and how to divide its limited transmission power. Independent or poorly coordinated decisions can:

- Send multiple beams to a cell that does not have enough queued traffic to use their capacity.
- Cause multiple satellites to compete for service of the same cell while other cells wait.
- Cause co-channel interference when active links use overlapping RBs.
- Waste transmission power and spectrum or increase traffic backlog.
- Produce uneven service across ground cells.

**Problem objective:** coordinate satellite-to-cell assignment, beam activation, RB allocation, power allocation, and conflict handling to deliver queued traffic effectively while respecting visibility and resource constraints and monitoring interference, contention, backlog, and fairness.

## 3. Goals and project scope

The project goals are to:

1. Simulate a multi-shell LEO downlink network with time-varying satellite geometry and ground traffic.
2. Enforce beam, RB, power, and visibility constraints.
3. Resolve simultaneous requests to the same cell consistently and reproducibly.
4. Model downlink signal strength, co-channel interference, SINR, achievable rate, and queue-capped delivery.
5. Implement a cooperative MAPPO learning pipeline with centralized training and decentralized execution (CTDE).
6. Compare the learned policy with Random, Greedy, and Rule-Based schedulers.
7. Support automated tests, command-line execution, saved experiment outputs, and interactive demonstration.

The current simulator is a research model with simplified orbital and geographic assumptions. It is not a flight-control system, an operational satellite network, or a full propagation/ephemeris simulator.

## 4. System configuration

The default configuration is in [`configs/default.yaml`](../configs/default.yaml); experiment scenarios are in [`configs/experiment.yaml`](../configs/experiment.yaml).

| Parameter | Default value |
|---|---:|
| Orbital shells | 2 |
| Satellites | 8 (4 per shell) |
| Shell altitudes | 550 km and 1100 km |
| Shell inclinations | 53° and 70° |
| Ground cells | 20 |
| Beams per satellite | 2 (16 total) |
| Resource blocks | 8 |
| Bandwidth per RB | 25 MHz (200 MHz total) |
| Carrier frequency | 20 GHz (Ka-band) |
| Slot duration / episode length | 1 second / 100 slots |
| Minimum visibility elevation | 25° |
| Maximum satellite transmit power | 40 W per satellite |
| Traffic hotspots | 25% of cells |
| Mean hotspot / regular arrival rates | 12 / 3.5 Mbps |
| Maximum cell queue capacity | 200 Mb |
| Maximum retained beams per cell | 2 |

These are simulation parameters, not measurements of a deployed constellation. Training and evaluation options, including the default evaluation seeds, are configured in the YAML files.

## 5. System model

### 5.1 Satellite geometry and visibility

The simulator advances satellites in two orbital shells and accounts for Earth rotation and line-of-sight elevation when determining which cells each satellite can serve. A cell is eligible for a satellite when its elevation is at least the configured minimum elevation angle. The model uses a localized set of ground cells around 40° N rather than a worldwide user distribution.

### 5.2 Traffic and queues

Each ground cell has a finite backlog. Arrivals are generated from the configured traffic distribution, including Poisson arrivals and optional burst behavior. A slot serves no more data than is pending in the cell's queue. In simplified form, the queue update is:

\[
Q_c(t+1) = \min\left(Q_{\max}, \max(0, Q_c(t)-x_c(t)) + A_c(t)\right)
\]

where \(Q_c(t)\) is the backlog for cell \(c\), \(x_c(t)\) is successfully delivered traffic during the slot, \(A_c(t)\) is new traffic, and \(Q_{\max}\) is the queue capacity.

### 5.3 Channel, interference, and delivery

The physical layer calculates slant range and elevation-dependent channel effects, free-space path loss, atmospheric attenuation, antenna gain including off-axis beam roll-off, received power, and thermal noise. Active retained links using overlapping RBs can contribute co-channel interference. The receiver SINR feeds a Shannon-style achievable-rate calculation with an implementation-efficiency factor. Actual delivered traffic is capped at pending backlog. Requests rejected by the conflict resolver do not transmit.

The implementation is intended for controlled simulation and comparison. Its physical assumptions should not be interpreted as a complete certification-grade Ka-band link budget.

### 5.4 Conflict resolution

For each cell \(c\), the maximum number of retained simultaneous requests is backlog-dependent:

\[
M_c(t) = \mathbf{1}[Q_c(t)>0]\,
\min\left(M_{c,\max}, \left\lceil Q_c(t)/Q_{\mathrm{seg}}\right\rceil\right)
\]

Eligible requests are scored using configured channel, power, queue, and delay terms. Requests are sorted deterministically, using satellite and beam identifiers to break ties. Requests above the admission limit are rejected. This makes request accounting reproducible and prevents rejected beams from contributing transmission or interference.

### 5.5 Resource constraints

Each satellite has a limited beam count and transmit power budget. Its beams use non-overlapping RB chunks, and beam power shares are constrained by the satellite's total budget, including reserved power. Dynamic masks prevent selecting invisible cells and RB chunks that conflict with already allocated chunks on the same satellite.

## 6. Proposed learning method

### 6.1 MAPPO with CTDE

The learning system uses Multi-Agent Proximal Policy Optimization (MAPPO) in a centralized-training, decentralized-execution setup. Satellite actors make decisions from their local observations; a centralized critic estimates value from global state during training. Actor parameters are shared across agents as implemented in the project.

### 6.2 Autoregressive actor

For each beam, the policy produces a sequence of dependent decisions:

1. Select a ground cell (or leave the beam inactive), subject to visibility masking.
2. Select an RB chunk, subject to intra-satellite RB exclusivity.
3. Select a continuous power share, parameterized by a Beta distribution.

Conditioning later choices on earlier ones represents the structure of the joint action while masks exclude infeasible choices.

### 6.3 Training and credit assignment

The training pipeline collects multi-agent rollouts, calculates generalized advantage estimates (GAE), normalizes advantages, and updates the actor and critic using PPO objectives, value loss, entropy regularization, and gradient clipping.

When enabled, counterfactual marginal beam credit compares the utility of the original slot with a replay in which an individual beam is deactivated:

\[
C_{i,k}(t) = U(\mathbf{a}(t)) - U(\mathbf{a}_{-i,k}(t))
\]

The existing environment tests are reported to check that this counterfactual evaluation does not mutate live environment state. This mechanism is intended to provide more specific credit for individual beam decisions; its presence does not itself establish improved policy performance.

## 7. Software architecture and project files

| Area | Responsibility |
|---|---|
| [`environment/`](../environment/) | Satellite network, traffic, channel, resource constraints, conflict resolution, transmission, and the multi-agent environment |
| [`agents/`](../agents/) | Action masks, observation encoding, autoregressive actor, centralized critic, and MAPPO agent |
| [`training/`](../training/) | Rollout collection, buffer, GAE/advantages, checkpoint handling, and trainer |
| [`baselines/`](../baselines/) | Random, Greedy, and Rule-Based schedulers |
| [`evaluation/`](../evaluation/) | Metrics, policy evaluation, baseline comparison, ablation, and plots |
| [`tests/`](../tests/) | Unit and integration tests for the environment, agents, resources, training, and evaluation |
| [`configs/`](../configs/) | Default network/training settings and experimental scenarios |
| [`checkpoints/`](../checkpoints/) | Existing trained model checkpoint |
| [`logs/`](../logs/) | Recorded simulation execution log |
| [`app.py`](../app.py) | Streamlit interactive dashboard |
| [`main.py`](../main.py) | Unified command-line entry point |
| [`requirements.txt`](../requirements.txt) | Python package requirements |

Within the environment, a slot follows this sequence: update geometry and visibility; encode observations and masks; select joint actions; validate resources; resolve cell conflicts; calculate transmissions, interference, SINR, and delivered traffic; update queues; calculate rewards and metrics; and advance the episode.

## 8. Engineering work recorded for the project

The existing implementation and verification reports document the following work and fixes:

- Implemented the multi-shell satellite and ground-cell traffic simulation, including dynamic queues and visibility masks.
- Added resource validation for beam limits, intra-satellite RB exclusivity, and power allocation.
- Implemented deterministic backlog-dependent conflict admission and request accounting.
- Implemented the channel/transmission path with interference, SINR, rate calculation, and queue-capped delivery.
- Added per-satellite observations and shell-level aggregate information for coordination.
- Implemented the autoregressive actor, centralized critic, MAPPO update loop, rollout buffer, GAE, and checkpoint support.
- Added counterfactual replay for per-beam marginal utility evaluation.
- Added Random, Greedy, and Rule-Based schedulers, evaluation metrics, benchmark comparison, ablation support, and plots.
- Added a command-line simulator/trainer/evaluator and a five-tab Streamlit dashboard.
- Resolved a reported circular import, added a missing PyTorch functional import, corrected ground-region/orbit alignment, and improved simulation telemetry.
- Improved Windows execution handling, including Matplotlib cache location and UTF-8 output.
- Connected the training `--episodes` option to the configured episode length and handled standard deviation for single-seed evaluation.
- Added checkpoint protection so a lower-evaluation-reward run does not replace the existing best checkpoint.

This list summarizes work recorded in the existing audit and verification documents. It is not a claim that all those changes were made during the preparation of this consolidated document.

## 9. Installation and execution

Run commands from the project root (the directory containing `main.py` and `requirements.txt`). Python 3.10+ is described in the README; the prior verification reports used Python 3.12. To install dependencies in a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, invoke the virtual-environment interpreter directly, for example `.\.venv\Scripts\python.exe -m pytest tests/ -v`.

### Common commands

```powershell
# Run all automated tests
python -m pytest tests/ -v

# Simulate one episode using the default rule-based policy
python main.py --mode simulate --episodes 1

# Run the trained checkpoint in the simulator
python main.py --mode simulate --policy trained --episodes 1

# Evaluate the existing checkpoint across configured seeds
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt

# Compare Random, Greedy, Rule-Based, and MAPPO policies
python main.py --mode compare --seeds 42 101

# Run a component ablation
python main.py --mode ablation --seeds 42 101

# Train for 128 scheduling slots
python main.py --mode train --slots 128 --eval_interval 128

# Launch the interactive dashboard
python -m streamlit run app.py
```

The CLI also supports `--mode all` to run a training, evaluation, comparison, and ablation sequence. Additional options include `--config`, `--checkpoint`, `--episodes`, `--slots`, `--policy`, `--eval_interval`, and `--seeds`. See [`main.py`](../main.py) for the available arguments.

### Generated outputs

Depending on the mode, the application writes training logs and curves, baseline comparison data and plots, ablation summaries and plots, checkpoints, and simulation logs. The default destinations are configured in `configs/default.yaml`; reported filenames include:

- `results/training_log.csv` and `results/training_curves.png`
- `results/baseline_comparison_raw.csv`, `results/baseline_comparison_summary.csv`, and `results/baseline_comparison.png`
- `results/ablation_study_summary.csv` and `results/ablation_study.png`
- `logs/simulation_execution.log`
- `checkpoints/best_model.pt`

Generated results may differ with configuration, code revision, checkpoint, and random seeds.

## 10. Verification record

The existing [`docs/REVIEW_2_VERIFICATION_REPORT.md`](REVIEW_2_VERIFICATION_REPORT.md) records a Windows/Python 3.12 verification run with the following outcomes:

| Verification area | Recorded outcome |
|---|---|
| Automated tests | 30 passed, 0 failed |
| Simulation demonstration | Completed; telemetry saved |
| Checkpoint inference | Existing checkpoint loaded and produced actions |
| Multi-seed evaluation | Completed |
| Baseline comparison | Completed and outputs generated |
| Ablation run | Completed and outputs generated |
| Short training run | Completed and training outputs generated |
| Streamlit dashboard | Reported to initialize its five tabs |

These are outcomes stated in the repository's verification report. The present documentation-only task did not rerun tests, training, benchmarks, or the dashboard. Use the commands above to reproduce the checks in the current Python environment.

## 11. Recorded benchmark and interpretation

The benchmark figures below are transcribed from the existing execution guide, which reports a five-seed comparison using seeds 42, 101, 202, 303, and 404:

| Policy | Delivered traffic (Mb) | Throughput (Mbps) | Final backlog (Mb) | Retention | Jain fairness | Interference (pW) |
|---|---:|---:|---:|---:|---:|---:|
| Random | 13650.9 ± 229.1 | 136.51 ± 2.29 | 548.0 ± 234.5 | 87.3% ± 0.2% | 0.194 ± 0.006 | 97.35 ± 11.16 |
| Greedy | 14036.2 ± 216.5 | 140.36 ± 2.17 | 312.3 ± 14.0 | 91.1% ± 0.6% | 0.269 ± 0.000 | 739.67 ± 30.88 |
| Rule-Based | **14075.2 ± 175.8** | **140.75 ± 1.76** | **273.3 ± 26.7** | **98.8% ± 0.5%** | **0.400 ± 0.003** | **192.03 ± 4.38** |
| ShellMean-MAPPO (epoch 4) | 6004.1 ± 486.6 | 60.04 ± 4.87 | 2309.4 ± 120.5 | 67.7% ± 0.7% | 0.198 ± 0.006 | 336.25 ± 60.74 |

**Interpretation:** In the recorded short-horizon comparison, the Rule-Based scheduler performs best on delivered traffic, backlog, retention, and fairness. The Greedy scheduler's reported interference is much higher. The existing MAPPO checkpoint is described as early-stage training (epoch 4) and does not outperform the Rule-Based baseline; extended training and fresh evaluation are needed before making stronger claims about the learned policy.

**Reproducibility note:** Existing reports contain differing baseline means and slot-runtime figures, and they report inconsistent observation/global-state dimensions. This consolidated document therefore does not assert a specific state dimension or runtime and does not combine values from those tables. Treat the table above as a historical report, not a guaranteed result. Re-run the comparison with a documented code revision, configuration, checkpoint, and seeds when preparing a final scientific result.

## 12. Current limitations and risks to interpretation

1. **Learning is not converged:** The available checkpoint is early-stage and is reported to underperform the Rule-Based policy. Network operation and passing tests are not evidence of policy convergence.
2. **Simplified orbital propagation:** The model uses simplified orbital kinematics rather than a full operational SGP4/TLE ephemeris workflow.
3. **Localized geography:** Ground cells are concentrated in a mid-latitude region rather than distributed globally.
4. **Simplified propagation assumptions:** Atmospheric and antenna models do not replace detailed, scenario-specific engineering link budgets or measurement-based channel models.
5. **Historical benchmark disagreement:** Documentation revisions report different baseline summaries. New experiments should save raw results and record exact code, configuration, checkpoint, and seed details.
6. **Simulation scope:** Results characterize this configured simulator only; they do not establish operational satellite-network performance.

## 13. Recommended next steps

- Train for a substantially longer, recorded budget and compare multiple checkpoints using fixed evaluation seeds.
- Re-run all baselines and ablations from the same code revision and traffic traces; preserve raw CSVs and record the exact config and checkpoint.
- Report uncertainty and define consistent metric units and aggregation procedures.
- Expand the constellation and ground-cell scenarios, and test traffic and interference sensitivity.
- Improve orbital/propagation realism where required, for example by adding validated ephemerides and richer weather/rain-fade assumptions.
- Keep claims aligned with measured evidence; specifically, claim MAPPO superiority only if replicated evaluations support it.

## 14. Conclusion

ShellMean-MAPPO is a modular simulation and experimentation framework for conflict-aware downlink resource allocation in multi-shell LEO networks. It combines a configurable network environment, physical transmission and conflict models, MAPPO with autoregressive masked actions, multiple baseline schedulers, training/evaluation tools, tests, and an interactive dashboard. The available project records report successful execution and 30 passing tests, while also showing that the current learned checkpoint remains behind the Rule-Based baseline. The project is a solid implementation and demonstration foundation; further reproducible training and evaluation are needed to establish the research performance of the learned policy.

## 15. Related project documentation

This is the consolidated end-to-end project document. For detailed subsystem-level or audit material, see:

- [`README.md`](../README.md) — project overview and quick start.
- [`README_EXECUTION.md`](../README_EXECUTION.md) — Windows execution and demonstration guide.
- [`technical_report.md`](technical_report.md) — technical model and research architecture.
- [`IMPLEMENTATION_STATUS_AUDIT.md`](IMPLEMENTATION_STATUS_AUDIT.md) — implementation audit.
- [`REVIEW_2_VERIFICATION_REPORT.md`](REVIEW_2_VERIFICATION_REPORT.md) — historical execution and test record.
