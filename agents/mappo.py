"""ShellMean-MAPPO Agent tying together Autoregressive Actor and Centralized Critic."""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.actor import AutoregressiveActor
from agents.critic import CentralizedCritic
from agents.action_masks import ActionMaskManager


class ShellMeanMAPPO(nn.Module):
    """
    Complete Multi-Agent PPO framework with:
    - Shared autoregressive actor across satellites
    - Centralized value critic
    - Action masking for cell visibility and intra-satellite RB exclusivity
    - Clipped PPO surrogate objective and entropy regularization
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.device = device if device is not None else torch.device("cpu")

        mappo_cfg = config.get("mappo", {})
        self.actor_lr = float(mappo_cfg.get("actor_lr", 0.0003))
        self.critic_lr = float(mappo_cfg.get("critic_lr", 0.001))
        self.clip_param = float(mappo_cfg.get("clip_param", 0.2))
        self.value_loss_coef = float(mappo_cfg.get("value_loss_coef", 0.5))
        self.entropy_coef = float(mappo_cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(mappo_cfg.get("max_grad_norm", 0.5))

        self.actor = AutoregressiveActor(config).to(self.device)
        self.critic = CentralizedCritic(config).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

        self.mask_mgr = self.actor.mask_mgr
        self.beams_per_satellite = self.actor.beams_per_satellite

    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        action_masks: Dict[str, Dict[str, Any]],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate actions for all satellites during environment rollout or evaluation.

        Returns:
            Tuple of:
                - actions_dict (for env.step)
                - log_probs_dict
                - beam_log_probs_dict
                - action_arrays_dict (raw indices and continuous powers)
        """
        actions = {}
        log_probs = {}
        beam_log_probs = {}
        action_arrays = {}

        self.actor.eval()
        with torch.no_grad():
            for sat_key, obs_np in observations.items():
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
                sat_mask = action_masks.get(sat_key, {})

                sat_act_dict, tot_lp, b_lps = self.actor.forward_sample(
                    obs=obs_t,
                    action_masks=sat_mask,
                    deterministic=deterministic,
                )

                actions[sat_key] = sat_act_dict
                log_probs[sat_key] = float(tot_lp.item())
                beam_log_probs[sat_key] = b_lps.cpu().numpy()

                # Extract compact numerical action arrays
                c_acts = [sat_act_dict[f"beam_{b}"]["cell_id"] for b in range(self.beams_per_satellite)]
                r_acts = [sat_act_dict[f"beam_{b}"]["rb_action_idx"] for b in range(self.beams_per_satellite)]
                p_acts = [sat_act_dict[f"beam_{b}"]["power_share"] for b in range(self.beams_per_satellite)]

                action_arrays[sat_key] = {
                    "cell_actions": np.array(c_acts, dtype=np.int64),
                    "rb_actions": np.array(r_acts, dtype=np.int64),
                    "power_actions": np.array(p_acts, dtype=np.float32),
                }

        return actions, log_probs, beam_log_probs, action_arrays

    def get_value(self, global_state: np.ndarray) -> float:
        """Estimate state value V(S) for a single global state."""
        self.critic.eval()
        with torch.no_grad():
            s_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            v = self.critic(s_t)
            return float(v.item())

    def update(
        self,
        batch_obs: torch.Tensor,
        batch_cell_acts: torch.Tensor,
        batch_rb_acts: torch.Tensor,
        batch_pwr_acts: torch.Tensor,
        batch_cell_masks: torch.Tensor,
        batch_rb_masks: torch.Tensor,
        batch_old_log_probs: torch.Tensor,
        batch_advantages: torch.Tensor,
        batch_global_states: torch.Tensor,
        batch_returns: torch.Tensor,
        batch_beam_credits: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Execute one PPO optimization step over a batch.

        Args:
            batch_obs: [B, obs_dim]
            batch_cell_acts: [B, K]
            batch_rb_acts: [B, K]
            batch_pwr_acts: [B, K]
            batch_cell_masks: [B, num_cells + 1]
            batch_rb_masks: [B, K, num_rb_actions]
            batch_old_log_probs: [B]
            batch_advantages: [B]
            batch_global_states: [B, global_state_dim]
            batch_returns: [B, 1]
            batch_beam_credits: Optional [B, K] counterfactual beam credit weights

        Returns:
            Dict containing loss metrics.
        """
        self.actor.train()
        self.critic.train()

        # 1. Evaluate current policy on batch
        new_tot_lp, new_b_lps, entropy = self.actor.evaluate_actions(
            obs=batch_obs,
            cell_actions=batch_cell_acts,
            rb_actions=batch_rb_acts,
            power_actions=batch_pwr_acts,
            cell_masks=batch_cell_masks,
            rb_masks=batch_rb_masks,
        )

        # 2. Actor Loss
        # Importance ratio
        ratio = torch.exp(new_tot_lp - batch_old_log_probs)

        # Integrate counterfactual credit if provided
        if batch_beam_credits is not None:
            # Credit-weighted advantage
            credit_weights = F.softmax(batch_beam_credits, dim=-1)  # [B, K]
            # Modulate advantage per beam
            eff_adv = batch_advantages.unsqueeze(-1) * (1.0 + credit_weights)
            # Beam-wise surrogate
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
        else:
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * entropy.mean()
        total_actor_loss = actor_loss + entropy_loss

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # 3. Critic Loss
        values = self.critic(batch_global_states)
        value_loss = 0.5 * F.mse_loss(values, batch_returns)
        total_critic_loss = self.value_loss_coef * value_loss

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        approx_kl = 0.5 * torch.mean((batch_old_log_probs - new_tot_lp) ** 2)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(value_loss.item()),
            "entropy": float(entropy.mean().item()),
            "approx_kl": float(approx_kl.item()),
            "mean_ratio": float(ratio.mean().item()),
        }

