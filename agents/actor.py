"""Autoregressive policy network for coordinated beam, RB, and power allocation."""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta

from agents.action_masks import ActionMaskManager


class AutoregressiveActor(nn.Module):
    """
    Autoregressive policy for multi-beam satellite agents.
    For each beam k sequentially:
    1. Selects ground cell a_{i,k}^{cell} (masked by visibility).
    2. Selects RB chunk (s, w) conditioned on cell choice (masked by intra-satellite exclusivity).
    3. Allocates transmit power share p_{i,k} conditioned on cell and RB selection.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        net_cfg = config.get("network", {})
        mappo_cfg = config.get("mappo", {})

        self.num_satellites = int(net_cfg.get("num_satellites", 8))
        self.num_shells = int(net_cfg.get("num_shells", 2))
        self.beams_per_satellite = int(net_cfg.get("beams_per_satellite", 2))
        self.num_ground_cells = int(net_cfg.get("num_ground_cells", 20))
        self.num_rbs = int(net_cfg.get("num_resource_blocks", 8))

        self.mask_mgr = ActionMaskManager(
            num_ground_cells=self.num_ground_cells,
            num_rbs=self.num_rbs,
            beams_per_satellite=self.beams_per_satellite,
        )
        self.num_cell_actions = self.num_ground_cells + 1  # 0: inactive, 1..N: cells
        self.num_rb_actions = self.mask_mgr.num_rb_actions

        # Feature dimensions
        self.local_feat_dim = self.num_satellites + self.num_shells + 6 + self.beams_per_satellite
        self.cells_feat_dim = 4 * self.num_ground_cells
        self.shell_sum_dim = 5 * self.num_shells
        self.obs_dim = self.local_feat_dim + self.cells_feat_dim + self.shell_sum_dim

        self.hidden_dim = int(mappo_cfg.get("hidden_dim", 128))

        # Shared base observation trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        # Autoregressive condition embeddings
        self.cell_embed = nn.Embedding(self.num_cell_actions, 32)
        self.rb_embed = nn.Embedding(self.num_rb_actions, 32)

        # Per-beam sequential decision heads
        self.cell_heads = nn.ModuleList()
        self.rb_heads = nn.ModuleList()
        self.power_heads = nn.ModuleList()

        for b in range(self.beams_per_satellite):
            # Condition dimension expands with prior beams' choices
            beam_ctx_dim = self.hidden_dim + b * (32 + 32 + 1)

            # 1. Cell selection head
            self.cell_heads.append(
                nn.Sequential(
                    nn.Linear(beam_ctx_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.num_cell_actions),
                )
            )

            # 2. RB selection head (conditioned on observation + beam context + chosen cell)
            self.rb_heads.append(
                nn.Sequential(
                    nn.Linear(beam_ctx_dim + 32, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.num_rb_actions),
                )
            )

            # 3. Power head: outputs Alpha and Beta concentration parameters for Beta distribution
            self.power_heads.append(
                nn.Sequential(
                    nn.Linear(beam_ctx_dim + 32 + 32, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 2),  # [alpha, beta]
                )
            )

    def forward_sample(
        self,
        obs: torch.Tensor,
        action_masks: Dict[str, Any],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """
        Sample coordinated autoregressive actions for a single satellite agent.

        Args:
            obs: 1D or 2D tensor of observations [obs_dim].
            action_masks: Mask dict containing 'cell_mask' and 'visible_cells'.
            deterministic: If True, selects mode / argmax actions.

        Returns:
            Tuple of:
                - action_dict: Dict suitable for environment step.
                - total_log_prob: Scalar tensor of total log probability.
                - beam_log_probs: Tensor of shape [beams_per_satellite] with individual beam log probs.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]

        h = self.trunk(obs)  # [1, hidden_dim]

        action_dict = {}
        total_log_prob = torch.tensor(0.0, device=obs.device)
        beam_log_probs = torch.zeros(self.beams_per_satellite, device=obs.device)

        occupied_rbs: set[int] = set()
        prior_context_list = []

        cell_mask_np = action_masks.get("cell_mask")
        if cell_mask_np is not None:
            cell_mask = torch.as_tensor(cell_mask_np, dtype=torch.bool, device=obs.device).unsqueeze(0)
        else:
            cell_mask = torch.ones((1, self.num_cell_actions), dtype=torch.bool, device=obs.device)

        for b_idx in range(self.beams_per_satellite):
            # Assemble context
            if prior_context_list:
                ctx = torch.cat([h] + prior_context_list, dim=-1)
            else:
                ctx = h

            # Step 1: Cell selection
            cell_logits = self.cell_heads[b_idx](ctx)  # [1, num_cell_actions]
            # Apply feasibility mask (-1e8 for invalid)
            masked_cell_logits = torch.where(cell_mask, cell_logits, torch.full_like(cell_logits, -1e8))
            cell_dist = Categorical(logits=masked_cell_logits)

            if deterministic:
                cell_act = torch.argmax(masked_cell_logits, dim=-1)
            else:
                cell_act = cell_dist.sample()

            cell_log_prob = cell_dist.log_prob(cell_act)
            cell_idx_val = int(cell_act.item())
            cell_emb = self.cell_embed(cell_act)  # [1, 32]

            # Step 2: RB selection
            rb_ctx = torch.cat([ctx, cell_emb], dim=-1)
            rb_mask_np = self.mask_mgr.get_rb_mask(occupied_rbs)
            rb_mask = torch.as_tensor(rb_mask_np, dtype=torch.bool, device=obs.device).unsqueeze(0)

            rb_logits = self.rb_heads[b_idx](rb_ctx)
            masked_rb_logits = torch.where(rb_mask, rb_logits, torch.full_like(rb_logits, -1e8))
            rb_dist = Categorical(logits=masked_rb_logits)

            if cell_idx_val == 0:
                # Inactive beam
                rb_act_val = 0
                rb_log_prob = torch.tensor(0.0, device=obs.device)
                s_rb, w_rb = 0, 0
            else:
                if deterministic:
                    rb_act = torch.argmax(masked_rb_logits, dim=-1)
                else:
                    rb_act = rb_dist.sample()
                rb_log_prob = rb_dist.log_prob(rb_act)
                rb_act_val = int(rb_act.item())
                s_rb, w_rb = self.mask_mgr.decode_rb_action(rb_act_val)
                # Register allocated RBs
                occupied_rbs.update(range(s_rb, s_rb + w_rb))

            rb_emb = self.rb_embed(torch.tensor([rb_act_val], device=obs.device))

            # Step 3: Power allocation
            pwr_ctx = torch.cat([rb_ctx, rb_emb], dim=-1)
            pwr_params = self.power_heads[b_idx](pwr_ctx)  # [1, 2]
            alpha = F.softplus(pwr_params[..., 0]) + 1.0
            beta = F.softplus(pwr_params[..., 1]) + 1.0
            pwr_dist = Beta(alpha, beta)

            if cell_idx_val == 0:
                p_val = 0.0
                pwr_log_prob = torch.tensor(0.0, device=obs.device)
            else:
                if deterministic:
                    # Beta distribution mode: (alpha - 1) / (alpha + beta - 2)
                    p_sample = (alpha - 1.0) / (alpha + beta - 2.0 + 1e-6)
                    p_sample = torch.clamp(p_sample, 0.05, 0.95)
                else:
                    p_sample = pwr_dist.sample()
                    p_sample = torch.clamp(p_sample, 0.01, 0.99)
                pwr_log_prob = pwr_dist.log_prob(p_sample).squeeze()
                p_val = float(p_sample.item())

            # Aggregate beam log-prob
            b_log_prob = cell_log_prob.squeeze() + rb_log_prob.squeeze() + pwr_log_prob.squeeze()
            beam_log_probs[b_idx] = b_log_prob
            total_log_prob = total_log_prob + b_log_prob

            action_dict[f"beam_{b_idx}"] = {
                "cell_id": cell_idx_val,
                "rb_start": s_rb,
                "rb_width": w_rb,
                "power_share": p_val,
                "rb_action_idx": rb_act_val,
            }

            # Update prior context for subsequent beams
            p_tensor = torch.tensor([[p_val]], device=obs.device)
            prior_context_list.extend([cell_emb, rb_emb, p_tensor])

        return action_dict, total_log_prob, beam_log_probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        cell_actions: torch.Tensor,
        rb_actions: torch.Tensor,
        power_actions: torch.Tensor,
        cell_masks: torch.Tensor,
        rb_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities and entropy for PPO updates.

        Args:
            obs: Batch of observations [batch_size, obs_dim].
            cell_actions: [batch_size, beams_per_sat].
            rb_actions: [batch_size, beams_per_sat].
            power_actions: [batch_size, beams_per_sat].
            cell_masks: [batch_size, num_cell_actions].
            rb_masks: [batch_size, beams_per_sat, num_rb_actions].

        Returns:
            Tuple of:
                - total_log_probs: [batch_size]
                - beam_log_probs: [batch_size, beams_per_sat]
                - entropy: [batch_size]
        """
        batch_size = obs.shape[0]
        h = self.trunk(obs)

        total_log_probs = torch.zeros(batch_size, device=obs.device)
        beam_log_probs = torch.zeros((batch_size, self.beams_per_satellite), device=obs.device)
        total_entropy = torch.zeros(batch_size, device=obs.device)

        prior_context_list = []

        for b_idx in range(self.beams_per_satellite):
            if prior_context_list:
                ctx = torch.cat([h] + prior_context_list, dim=-1)
            else:
                ctx = h

            # 1. Cell evaluation
            cell_logits = self.cell_heads[b_idx](ctx)
            c_mask = cell_masks[:, b_idx] if cell_masks.dim() == 3 else cell_masks
            masked_cell_logits = torch.where(c_mask, cell_logits, torch.full_like(cell_logits, -1e8))
            cell_dist = Categorical(logits=masked_cell_logits)

            c_act = cell_actions[:, b_idx]
            cell_lp = cell_dist.log_prob(c_act)
            cell_ent = cell_dist.entropy()
            cell_emb = self.cell_embed(c_act)

            # 2. RB evaluation
            rb_ctx = torch.cat([ctx, cell_emb], dim=-1)
            rb_logits = self.rb_heads[b_idx](rb_ctx)
            r_mask = rb_masks[:, b_idx]
            masked_rb_logits = torch.where(r_mask, rb_logits, torch.full_like(rb_logits, -1e8))
            rb_dist = Categorical(logits=masked_rb_logits)

            r_act = rb_actions[:, b_idx]
            rb_lp = rb_dist.log_prob(r_act)
            rb_ent = rb_dist.entropy()
            rb_emb = self.rb_embed(r_act)

            # 3. Power evaluation
            pwr_ctx = torch.cat([rb_ctx, rb_emb], dim=-1)
            pwr_params = self.power_heads[b_idx](pwr_ctx)
            alpha = F.softplus(pwr_params[..., 0]) + 1.0
            beta = F.softplus(pwr_params[..., 1]) + 1.0
            pwr_dist = Beta(alpha, beta)

            p_act = torch.clamp(power_actions[:, b_idx], 0.001, 0.999)
            pwr_lp = pwr_dist.log_prob(p_act)
            pwr_ent = pwr_dist.entropy()

            # Zero out log prob and entropy for inactive beams
            active_mask = (c_act > 0).float()
            b_lp = cell_lp + (rb_lp + pwr_lp) * active_mask
            b_ent = cell_ent + (rb_ent + pwr_ent) * active_mask

            beam_log_probs[:, b_idx] = b_lp
            total_log_probs = total_log_probs + b_lp
            total_entropy = total_entropy + b_ent

            p_tensor = p_act.unsqueeze(-1)
            prior_context_list.extend([cell_emb, rb_emb, p_tensor])

        return total_log_probs, beam_log_probs, total_entropy

