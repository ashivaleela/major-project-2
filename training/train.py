"""MAPPO training loop, mini-batch updates, and convergence tracking."""

from __future__ import annotations
import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

from environment.leo_env import MultiShellLEOEnv
from agents.mappo import ShellMeanMAPPO
from training.buffer import RolloutBuffer
from training.rollout import collect_rollout
from training.advantages import normalize_advantages
from training.checkpoint import save_checkpoint, load_checkpoint


class MAPPOTrainer:
    """Orchestrates multi-agent PPO training, logging, and evaluation."""

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        self.config = config
        self.device = device if device is not None else torch.device("cpu")

        mappo_cfg = config.get("mappo", {})
        net_cfg = config.get("network", {})
        exp_cfg = config.get("experiment", {})

        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.rollout_steps = int(mappo_cfg.get("rollout_steps", 64))
        self.mini_batch_size = int(mappo_cfg.get("mini_batch_size", 64))
        self.ppo_epochs = int(mappo_cfg.get("ppo_epochs", 4))

        self.env = MultiShellLEOEnv(config, seed=int(exp_cfg.get("seed", 42)))
        self.eval_env = MultiShellLEOEnv(config, seed=int(exp_cfg.get("seed", 42)) + 1000)

        self.agent = ShellMeanMAPPO(config, device=self.device)

        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            num_satellites=self.num_satellites,
            obs_dim=self.env.obs_dim,
            global_state_dim=self.env.global_state_dim,
            beams_per_satellite=self.beams_per_satellite,
            num_cell_actions=self.agent.actor.num_cell_actions,
            num_rb_actions=self.agent.mask_mgr.num_rb_actions,
        )

        self.history: List[Dict[str, Any]] = []
        self.best_eval_reward = -float("inf")

    def step_evaluation(
        self,
        observations: Dict[str, np.ndarray],
        action_masks: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate evaluation actions using deterministic actor inference."""
        actions, _, _, _ = self.agent.get_actions(
            observations=observations,
            action_masks=action_masks,
            deterministic=True,
        )
        return actions

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        action_masks: Dict[str, Dict[str, Any]],
        env: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Wrapper for evaluate_policy pipeline compatibility."""
        return self.step_evaluation(observations, action_masks)

    def train(
        self,
        total_slots: int = 1000,
        eval_interval: int = 200,
        checkpoint_path: str = "checkpoints/best_model.pt",
        use_counterfactual_credit: bool = True,
    ) -> pd.DataFrame:
        """
        Execute full training procedure.

        Args:
            total_slots: Total environment interaction steps.
            eval_interval: Steps between evaluation rounds.
            checkpoint_path: Filepath to save best model checkpoint.
            use_counterfactual_credit: Enable counterfactual marginal beam credit.

        Returns:
            DataFrame containing training logs.
        """
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        if os.path.exists(checkpoint_path):
            try:
                prior = torch.load(checkpoint_path, map_location="cpu")
                if "metrics" in prior and "eval_reward" in prior["metrics"]:
                    self.best_eval_reward = float(prior["metrics"]["eval_reward"])
            except Exception:
                pass
        print(f"Starting ShellMean-MAPPO training for {total_slots} slots (Prior best reward: {self.best_eval_reward:.2f})...")

        slots_trained = 0
        iteration = 0
        curr_obs, curr_info = self.env.reset()

        while slots_trained < total_slots:
            iteration += 1
            t_start = time.perf_counter()

            # 1. Collect rollout
            curr_obs, curr_info, telem = collect_rollout(
                env=self.env,
                agent=self.agent,
                buffer=self.buffer,
                use_counterfactual_credit=use_counterfactual_credit,
                current_obs=curr_obs,
                current_info=curr_info,
            )
            slots_trained += self.rollout_steps

            # 2. Prepare flat satellite training batch
            # Normalize advantages
            norm_adv = normalize_advantages(self.buffer.advantages)

            # Flatten across rollout_steps and num_satellites
            n_slots = self.rollout_steps
            n_sats = self.num_satellites
            total_samples = n_slots * n_sats

            flat_obs = self.buffer.obs.reshape((total_samples, self.env.obs_dim))
            flat_cell_acts = self.buffer.cell_actions.reshape((total_samples, self.beams_per_satellite))
            flat_rb_acts = self.buffer.rb_actions.reshape((total_samples, self.beams_per_satellite))
            flat_pwr_acts = self.buffer.power_actions.reshape((total_samples, self.beams_per_satellite))
            flat_cell_masks = self.buffer.cell_masks.reshape((total_samples, self.agent.actor.num_cell_actions))
            flat_rb_masks = self.buffer.rb_masks.reshape(
                (total_samples, self.beams_per_satellite, self.agent.mask_mgr.num_rb_actions)
            )
            flat_old_lp = self.buffer.log_probs.reshape(total_samples)

            # Replicate slot-level global state, advantages, and returns across all satellites
            flat_global_states = np.repeat(self.buffer.global_states, n_sats, axis=0)
            flat_adv = np.repeat(norm_adv, n_sats, axis=0)
            flat_returns = np.repeat(self.buffer.returns, n_sats, axis=0).reshape((total_samples, 1))
            flat_credits = self.buffer.beam_credits.reshape((total_samples, self.beams_per_satellite))

            # Convert to PyTorch tensors
            t_obs = torch.as_tensor(flat_obs, dtype=torch.float32, device=self.device)
            t_cell_acts = torch.as_tensor(flat_cell_acts, dtype=torch.int64, device=self.device)
            t_rb_acts = torch.as_tensor(flat_rb_acts, dtype=torch.int64, device=self.device)
            t_pwr_acts = torch.as_tensor(flat_pwr_acts, dtype=torch.float32, device=self.device)
            t_cell_masks = torch.as_tensor(flat_cell_masks, dtype=torch.bool, device=self.device)
            t_rb_masks = torch.as_tensor(flat_rb_masks, dtype=torch.bool, device=self.device)
            t_old_lp = torch.as_tensor(flat_old_lp, dtype=torch.float32, device=self.device)
            t_adv = torch.as_tensor(flat_adv, dtype=torch.float32, device=self.device)
            t_global_states = torch.as_tensor(flat_global_states, dtype=torch.float32, device=self.device)
            t_returns = torch.as_tensor(flat_returns, dtype=torch.float32, device=self.device)
            t_credits = (
                torch.as_tensor(flat_credits, dtype=torch.float32, device=self.device)
                if use_counterfactual_credit
                else None
            )

            # 3. PPO Epochs and Mini-batch optimization
            indices = np.arange(total_samples)
            epoch_losses = []

            for _ in range(self.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, total_samples, self.mini_batch_size):
                    batch_idx = indices[start : start + self.mini_batch_size]

                    loss_info = self.agent.update(
                        batch_obs=t_obs[batch_idx],
                        batch_cell_acts=t_cell_acts[batch_idx],
                        batch_rb_acts=t_rb_acts[batch_idx],
                        batch_pwr_acts=t_pwr_acts[batch_idx],
                        batch_cell_masks=t_cell_masks[batch_idx],
                        batch_rb_masks=t_rb_masks[batch_idx],
                        batch_old_log_probs=t_old_lp[batch_idx],
                        batch_advantages=t_adv[batch_idx],
                        batch_global_states=t_global_states[batch_idx],
                        batch_returns=t_returns[batch_idx],
                        batch_beam_credits=t_credits[batch_idx] if t_credits is not None else None,
                    )
                    epoch_losses.append(loss_info)

            t_elapsed = time.perf_counter() - t_start

            # Aggregate losses
            mean_loss = {k: float(np.mean([l[k] for l in epoch_losses])) for k in epoch_losses[0].keys()}

            # 4. Periodic Evaluation
            eval_metrics = {}
            if slots_trained % eval_interval < self.rollout_steps or slots_trained >= total_slots:
                from evaluation.evaluate import evaluate_policy
                eval_seeds = self.config.get("experiment", {}).get("eval_seeds", [42, 101])[:3]
                eval_res = evaluate_policy(self.eval_env, self, "ShellMean_MAPPO", seeds=eval_seeds)
                eval_reward = eval_res["means"]["cumulative_reward"]
                eval_throughput = eval_res["means"]["mean_throughput_mbps"]
                eval_metrics = {
                    "eval_reward": eval_reward,
                    "eval_throughput": eval_throughput,
                }

                # Save best model
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    save_checkpoint(
                        agent=self.agent,
                        filepath=checkpoint_path,
                        epoch=iteration,
                        metrics=eval_metrics,
                        config=self.config,
                    )
                    print(f"[*] New best model saved at {checkpoint_path} (Reward: {eval_reward:.2f})")

            log_entry = {
                "iteration": iteration,
                "slots_trained": slots_trained,
                "rollout_reward": telem["mean_reward"],
                "rollout_served_mb": telem["mean_served_mb"],
                "rollout_retention": telem["mean_retention_rate"],
                "actor_loss": mean_loss["actor_loss"],
                "critic_loss": mean_loss["critic_loss"],
                "entropy": mean_loss["entropy"],
                "approx_kl": mean_loss["approx_kl"],
                "step_time_sec": t_elapsed,
                **eval_metrics,
            }
            self.history.append(log_entry)

            print(
                f"Iter {iteration:03d} | Slots: {slots_trained:04d}/{total_slots} | "
                f"Rew: {telem['mean_reward']:.2f} | Served: {telem['mean_served_mb']:.1f} Mb | "
                f"Ret: {telem['mean_retention_rate']:.1f}% | "
                f"Loss(A/C): {mean_loss['actor_loss']:.3f}/{mean_loss['critic_loss']:.3f} | "
                f"Time: {t_elapsed:.2f}s"
            )

        # Save final checkpoint if best not yet written
        if not os.path.exists(checkpoint_path):
            save_checkpoint(
                agent=self.agent,
                filepath=checkpoint_path,
                epoch=iteration,
                metrics={"final": True},
                config=self.config,
            )

        df_history = pd.DataFrame(self.history)
        return df_history

