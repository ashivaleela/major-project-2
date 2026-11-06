"""Standalone deterministic conflict resolution module for multi-satellite beam collisions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ServiceRequest:
    """A downlink service request submitted by a satellite beam."""
    sat_id: int
    shell_id: int
    beam_id: int
    cell_id: int  # 0: Inactive, 1..N: cell index (cell_id - 1)
    power_share: float
    rb_start: int
    rb_width: int
    channel_gain: float
    propagation_delay_sec: float
    queue_backlog_mb: float = 0.0

    @property
    def target_cell_index(self) -> Optional[int]:
        return (self.cell_id - 1) if self.cell_id > 0 else None


@dataclass
class ResolutionOutcome:
    """Structured decision output of the conflict resolution process."""
    retained_requests: List[ServiceRequest] = field(default_factory=list)
    rejected_requests: List[ServiceRequest] = field(default_factory=list)
    rejection_reasons: Dict[tuple[int, int], str] = field(default_factory=dict)
    resolution_scores: Dict[tuple[int, int], float] = field(default_factory=dict)
    cell_retention_limits: Dict[int, int] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


class ConflictResolver:
    """
    Deterministic conflict resolver.
    Evaluates simultaneous requests targeting the same ground cell, computes
    backlog-dependent retention limits, ranks requests via multi-attribute
    resolution scoring, and enforces deterministic retention/rejection.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize conflict resolver parameters.

        Args:
            config: Full configuration dictionary containing 'conflict_resolution' section.
        """
        self.config = config
        cr_cfg = config.get("conflict_resolution", {})
        net_cfg = config.get("network", {})

        self.m_c_max = int(cr_cfg.get("m_c_max", 2))
        self.q_seg_mb = float(cr_cfg.get("q_seg_mb", 12.0))
        self.w_channel = float(cr_cfg.get("weight_channel", 1.0))
        self.w_power = float(cr_cfg.get("weight_power", 0.8))
        self.w_queue = float(cr_cfg.get("weight_queue", 1.2))
        self.w_delay = float(cr_cfg.get("weight_delay", 0.5))

        self.num_cells = int(net_cfg.get("num_ground_cells", 20))
        self.max_queue_ref = float(config.get("traffic", {}).get("max_queue_capacity_mb", 200.0))
        self.max_delay_ref = 0.01  # 10 ms reference max propagation delay

    def compute_retention_limit(self, queue_backlog_mb: float) -> int:
        """
        Compute backlog-dependent retention limit:
            M_c(t) = 1[Q_c(t) > 0] * min(M_c_max, ceil(Q_c(t) / Q_seg))

        Args:
            queue_backlog_mb: Pending traffic in cell queue (Mb).

        Returns:
            Max number of simultaneous beams permitted to serve this cell (0 <= M_c <= M_c_max).
        """
        if queue_backlog_mb <= 0.0:
            return 0
        seg_count = int(np.ceil(queue_backlog_mb / max(0.1, self.q_seg_mb)))
        return int(min(self.m_c_max, max(1, seg_count)))

    def compute_resolution_score(
        self,
        request: ServiceRequest,
        mean_cell_channel_gain: float,
    ) -> float:
        """
        Compute multi-attribute resolution score S(i, k, c):
            S = w_h * (h / h_mean) + w_p * p + w_q * (Q / Q_max) - w_tau * (tau / tau_ref)

        Args:
            request: The service request.
            mean_cell_channel_gain: Average channel gain across candidates for normalization.

        Returns:
            Scalar resolution score.
        """
        norm_h = request.channel_gain / max(1e-25, mean_cell_channel_gain)
        norm_p = float(np.clip(request.power_share, 0.0, 1.0))
        norm_q = float(np.clip(request.queue_backlog_mb / self.max_queue_ref, 0.0, 1.0))
        norm_delay = float(np.clip(request.propagation_delay_sec / self.max_delay_ref, 0.0, 2.0))

        score = (
            self.w_channel * norm_h
            + self.w_power * norm_p
            + self.w_queue * norm_q
            - self.w_delay * norm_delay
        )
        return float(score)

    def resolve_conflicts(
        self,
        requests: List[ServiceRequest],
        queue_backlogs: np.ndarray,
        visibility_matrix: np.ndarray,
    ) -> ResolutionOutcome:
        """
        Execute deterministic conflict resolution.

        Processing Sequence:
        1. Collect all requests.
        2. Validate eligibility (reject inactive, invisible, or queue <= 0).
        3. Group eligible requests by target cell.
        4. Calculate retention limit M_c(t) for each cell.
        5. Score eligible requests.
        6. Deterministically rank by (score desc, sat_id asc, beam_id asc).
        7. Retain up to M_c(t) requests; reject remaining excess.
        8. Return structured outcome with complete accounting and logs.

        Args:
            requests: All submitted service requests across all satellites and beams.
            queue_backlogs: Array of shape [num_cells] with current queue sizes (Mb).
            visibility_matrix: Array of shape [num_satellites, num_cells] indicating visibility.

        Returns:
            ResolutionOutcome containing retained and rejected requests.
        """
        outcome = ResolutionOutcome()

        # Step 1 & 2: Filter eligibility
        eligible_requests_by_cell: Dict[int, List[ServiceRequest]] = {c: [] for c in range(self.num_cells)}

        for req in requests:
            req_key = (req.sat_id, req.beam_id)

            # Inactive request
            if req.cell_id <= 0:
                outcome.rejected_requests.append(req)
                outcome.rejection_reasons[req_key] = "Beam explicitly set to inactive."
                continue

            target_c = req.target_cell_index
            if target_c is None or target_c < 0 or target_c >= self.num_cells:
                outcome.rejected_requests.append(req)
                outcome.rejection_reasons[req_key] = f"Invalid target cell index {target_c}."
                continue

            # Visibility check
            if not visibility_matrix[req.sat_id, target_c]:
                outcome.rejected_requests.append(req)
                outcome.rejection_reasons[req_key] = f"Target cell {target_c} is not visible to sat {req.sat_id}."
                continue

            # Queue backlog check
            cell_q = float(queue_backlogs[target_c])
            req.queue_backlog_mb = cell_q
            if cell_q <= 0.0:
                outcome.rejected_requests.append(req)
                outcome.rejection_reasons[req_key] = f"Target cell {target_c} has zero pending traffic queue."
                continue

            # Request is eligible for competitive resolution
            eligible_requests_by_cell[target_c].append(req)

        # Step 3, 4, 5, 6, 7: Process each cell deterministically
        for c_id in sorted(eligible_requests_by_cell.keys()):
            cell_requests = eligible_requests_by_cell[c_id]
            cell_q = float(queue_backlogs[c_id])
            retention_limit = self.compute_retention_limit(cell_q)
            outcome.cell_retention_limits[c_id] = retention_limit

            if not cell_requests:
                continue

            # Compute normalization reference: mean channel gain of requesting satellites
            mean_gain = float(np.mean([r.channel_gain for r in cell_requests]))

            # Score each request
            scored_requests = []
            for req in cell_requests:
                score = self.compute_resolution_score(req, mean_gain)
                req_key = (req.sat_id, req.beam_id)
                outcome.resolution_scores[req_key] = score
                scored_requests.append((score, req.sat_id, req.beam_id, req))

            # Deterministic stable ranking:
            # -score (descending), sat_id (ascending), beam_id (ascending)
            scored_requests.sort(key=lambda item: (-item[0], item[1], item[2]))

            # Retain top M_c(t)
            retained_count = 0
            for score, s_id, b_id, req in scored_requests:
                req_key = (s_id, b_id)
                if retained_count < retention_limit:
                    outcome.retained_requests.append(req)
                    retained_count += 1
                    outcome.logs.append(
                        f"Retained sat {s_id} beam {b_id} for cell {c_id} (Score={score:.4f}, Rank={retained_count}/{retention_limit})."
                    )
                else:
                    outcome.rejected_requests.append(req)
                    outcome.rejection_reasons[req_key] = (
                        f"Rejected by conflict resolver: Cell {c_id} reached capacity limit "
                        f"{retention_limit} (Score={score:.4f})."
                    )
                    outcome.logs.append(
                        f"Rejected sat {s_id} beam {b_id} for cell {c_id} (Score={score:.4f}, Exceeded limit {retention_limit})."
                    )

        # Strict validation: Ensure every submitted request is either retained or rejected
        total_in = len(requests)
        total_out = len(outcome.retained_requests) + len(outcome.rejected_requests)
        assert total_in == total_out, f"ConflictResolver integrity error: {total_in} in != {total_out} out!"

        return outcome

