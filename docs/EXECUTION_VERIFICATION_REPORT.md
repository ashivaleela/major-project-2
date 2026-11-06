# ShellMean-MAPPO: Execution & Verification Audit Report

**Date of Verification:** October 29, 2026  
**Auditor / Role:** Senior AI Research Engineer & Verification Specialist  
**Project:** ShellMean-MAPPO: A Conflict-Aware Multi-Agent Reinforcement Learning Framework for Downlink Resource Allocation in Multi-Shell LEO Satellite Networks  
**Target Milestone:** Major Project Review 2  

---

### 1. Executive Inspection & Workspace State

A rigorous audit of the workspace at `C:\Users\ashiv\OneDrive\Documents\Major Project 1` was conducted prior to and during verification:

- **Working Directory:** `C:\Users\ashiv\OneDrive\Documents\Major Project 1`
- **Python Environment:** Python 3.14.0 (64-bit AMD64) and configured virtual environment at `.\venv` with system site-packages.
- **Dependencies Verified:** PyTorch (`2.15.0+cpu`), NumPy (`2.4.4`), Pandas (`3.0.2`), Matplotlib (`3.10.8`), PyYAML (`6.0.3`), Gymnasium (`1.4.0`), Streamlit (`1.66.1`), Pytest (`9.2.0`), Scipy (`1.17.1`).
- **Checkpoints Audited & Backed Up:**
  - `checkpoints/best_model.pt` (Verified: 2,918,273 bytes; PyTorch state_dict format with actor, critic, optimizers, epoch, and config).
  - `checkpoints/best_model_backup.pt` (Created safety backup).
  - `checkpoints/latest_model_backup.pt` (Created safety backup).
- **Execution Artifacts Verified:**
  - `results/training_log.csv` (Logged iterations 1–4, 512 scheduling slots).
  - `results/training_curves.png` (4-panel PNG figure).
  - `results/baseline_comparison_raw.csv` (20 completed runs across 5 seeds).
  - `results/baseline_comparison_summary.csv` (Aggregated metrics table).
  - `results/baseline_comparison.png` (4-panel comparison bar chart).
  - `results/ablation_study_summary.csv` (Ablation analysis table).
  - `results/ablation_study.png` (Ablation chart).
  - `logs/smoke_test_execution.log` (Full physical downlink slot telemetry log).

---

### 2. Issues Diagnosed & Resolved

During audit and execution verification, the following genuine engineering issues were identified, diagnosed, and resolved:

| Issue / Error | Root Cause | Resolution Applied | Verification Evidence |
|---|---|---|---|
| **Circular Import on Module Load** | `training/train.py` imported `evaluate_policy` from `evaluation.evaluate`, while `evaluation/__init__.py` loaded `compare_baselines.py`, which imported `MAPPOTrainer` from `training.train`. | Decoupled cyclic module dependency: moved `evaluate_policy` import inside the evaluation method block in `train.py` and lazy-imported `MAPPOTrainer` in comparison scripts. | `main.py` CLI runs cleanly in all modes (`simulate`, `train`, `evaluate`, `compare`, `ablation`). |
| **Missing `F` in `agents/mappo.py`** | `import torch.nn.functional as F` was omitted from top-level imports in `agents/mappo.py`. | Added `import torch.nn.functional as F` to `agents/mappo.py`. | `test_mappo.py` passed with 100% gradient step success. |
| **Ground Cell Zenith Alignment** | Initially, ground cells were centered at $(0, 0, R_E)$ (North Pole), which fell outside the $53^\circ$ and $70^\circ$ orbital inclination coverage cones. | Re-centered operational ground region around mid-latitude ($40^\circ\text{ N}$) and phased satellite anomalies to generate realistic multi-satellite visibility and competitive contention. | All satellites in both shells exhibit dynamic line-of-sight elevation angles and multi-satellite competition. |
| **Execution Policy in PowerShell** | Virtual environment activation script (`Activate.ps1`) may be blocked by restrictive Windows execution policies. | Configured `venv` with `--system-site-packages` and provided process-scoped bypass syntax (`powershell -ExecutionPolicy Bypass ...`). | Verified direct invocation via `.\venv\Scripts\python.exe` and activated shell. |
| **Missing Live Slot Physical Telemetry** | Default simulation output only showed episode-level aggregated numbers, without per-beam physical metrics required for Review 2 inspection. | Enhanced `main.py` `--mode simulate` to output comprehensive breakdown: visible cells, proposed beam requests, conflict resolution ranking, physical $P_{\text{tx}}$, $P_{\text{rx}}$, interference ($I$), SINR (dB), Shannon rate (Mbps), and queue updates. | Verified live output saved in `logs/smoke_test_execution.log`. |

---

### 3. Automated Test Suite Results

Test execution command:
```powershell
python -m pytest tests/ -v
```

**Results:** **30 passed in 15.46 seconds (0 failures, 0 errors, 0 skipped)**

```text
tests/test_actor.py::test_actor_sampling_and_mask_enforcement PASSED     [  3%]
tests/test_actor.py::test_actor_critic_gradient_flow PASSED              [  6%]
tests/test_conflict_resolver.py::test_empty_queue_retains_zero_requests PASSED [ 10%]
tests/test_conflict_resolver.py::test_retention_limit_not_exceeded PASSED [ 13%]
tests/test_conflict_resolver.py::test_deterministic_ranking_and_tie_breaking PASSED [ 16%]
tests/test_conflict_resolver.py::test_total_request_accounting PASSED    [ 20%]
tests/test_environment.py::test_env_reset PASSED                         [ 23%]
tests/test_environment.py::test_env_step_execution PASSED                [ 26%]
tests/test_environment.py::test_counterfactual_step_non_mutation PASSED  [ 30%]
tests/test_evaluation.py::test_random_policy_execution PASSED            [ 33%]
tests/test_evaluation.py::test_greedy_policy_execution PASSED            [ 36%]
tests/test_evaluation.py::test_rule_based_policy_execution PASSED        [ 40%]
tests/test_evaluation.py::test_multi_seed_evaluation PASSED              [ 43%]
tests/test_mappo.py::test_gae_and_advantage_normalization PASSED         [ 46%]
tests/test_mappo.py::test_mappo_rollout_and_update PASSED                [ 50%]
tests/test_mappo.py::test_checkpoint_saving_and_loading PASSED           [ 53%]
tests/test_observations.py::test_observation_dimensions_and_finite_values PASSED [ 56%]
tests/test_observations.py::test_nan_sanitation PASSED                   [ 60%]
tests/test_resources.py::test_invisible_cell_rejection PASSED            [ 63%]
tests/test_resources.py::test_intra_satellite_rb_exclusivity PASSED      [ 66%]
tests/test_resources.py::test_power_conservation PASSED                  [ 70%]
tests/test_resources.py::test_rb_overlap_calculation PASSED              [ 73%]
tests/test_traffic.py::test_ground_cell_queue_update PASSED              [ 76%]
tests/test_traffic.py::test_service_capped_at_queue PASSED               [ 80%]
tests/test_traffic.py::test_queue_capacity_limit PASSED                  [ 83%]
tests/test_traffic.py::test_traffic_model_reset_and_step PASSED          [ 86%]
tests/test_transmission.py::test_zero_interference_with_disjoint_rbs PASSED [ 90%]
tests/test_transmission.py::test_positive_interference_with_overlapping_rbs PASSED [ 93%]
tests/test_transmission.py::test_service_capped_strictly_at_backlog PASSED [ 96%]
tests/test_transmission.py::test_rejected_beams_deliver_zero PASSED      [100%]
```

---

### 4. Checkpoint Loading & Evaluation Audit

The checkpoint file `checkpoints/best_model.pt` was inspected and loaded into `ShellMeanMAPPO`:
- **Format:** PyTorch dictionary mapping containing `epoch` (int), `actor_state_dict` (OrderedDict), `critic_state_dict` (OrderedDict), `actor_optimizer_state_dict` (dict), `critic_optimizer_state_dict` (dict), `metrics` (dict), and `config` (dict).
- **Evaluation Command:** `python main.py --mode evaluate --checkpoint checkpoints/best_model.pt`
- **Output:**
  - Delivered Traffic: $6004.1 \pm 486.6\text{ Mb}$
  - Mean Throughput: $60.04 \pm 4.87\text{ Mbps}$
  - Episode Reward: $5332.2 \pm 498.7$
  - Retention Rate: $67.7\% \pm 0.7\%$
  - Conflict Rate: $32.3\% \pm 0.7\%$
  - Jain's Fairness Index: $0.198 \pm 0.006$
  - Average Co-Channel Interference: $336.25 \pm 60.74\text{ pW}$

---

### 5. Benchmark Performance Verification (Experiment A)

Evaluated across **5 independent evaluation seeds** (`[42, 101, 202, 303, 404]`):

| Algorithm | Delivered Traffic (Mb) | Throughput (Mbps) | Episode Reward | Final Backlog (Mb) | Retention Rate (%) | Conflict Rate (%) | Jain's Fairness | Interference (pW) | Slot Runtime (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **Random** | $13638.3 \pm 201.2$ | $136.38 \pm 2.01$ | $13417.1 \pm 201.1$ | $510.6 \pm 143.2$ | $86.7\% \pm 1.2\%$ | $13.3\% \pm 1.2\%$ | $0.190 \pm 0.005$ | $99.85 \pm 25.66$ | $0.54 \pm 0.00$ |
| **Greedy** | $13937.6 \pm 170.9$ | $139.38 \pm 1.71$ | $13775.6 \pm 171.8$ | $337.0 \pm 58.5$ | $90.9\% \pm 0.5\%$ | $9.1\% \pm 0.5\%$ | $0.267 \pm 0.004$ | $721.86 \pm 33.67$ | $0.31 \pm 0.02$ |
| **Rule-Based** | **$14014.8 \pm 172.5$** | **$140.15 \pm 1.73$** | **$13946.7 \pm 173.8$** | **$259.8 \pm 22.1$** | **$99.1\% \pm 0.4\%$** | **$0.9\% \pm 0.4\%$** | **$0.396 \pm 0.006$** | $212.99 \pm 30.06$ | $0.36 \pm 0.01$ |
| **ShellMean-MAPPO** | $6004.1 \pm 486.6$ | $60.04 \pm 4.87$ | $5332.2 \pm 498.7$ | $2309.4 \pm 81.8$ | $67.7\% \pm 0.7\%$ | $32.3\% \pm 0.7\%$ | $0.198 \pm 0.006$ | $336.25 \pm 60.74$ | $46.37 \pm 0.86$ |

---

### 6. Recommended Execution Order for Review 2

1. **Step 1:** Run unit & integration test verification:
   ```powershell
   python -m pytest tests/ -v
   ```
2. **Step 2:** Run live physical scheduling demonstration:
   ```powershell
   python main.py --mode simulate --episodes 1
   ```
3. **Step 3:** Evaluate saved neural policy checkpoint:
   ```powershell
   python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
   ```
4. **Step 4:** Launch interactive Streamlit dashboard:
   ```powershell
   streamlit run app.py
   ```

