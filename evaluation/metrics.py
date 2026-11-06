"""Comprehensive evaluation metrics for multi-shell LEO downlink resource allocation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class SlotMetrics:
    """Metrics recorded for a single scheduling slot t."""
    step: int
    team_reward: float
    served_traffic_mb: float
    throughput_mbps: float
    total_backlog_mb: float
    queue_growth_mb: float
    num_submitted: int
    num_retained: int
    num_rejected: int
    retention_rate: float
    conflict_rate: float
    interference_watts: float
    jains_fairness: float
    active_beam_ratio: float
    rb_utilization: float
    power_utilization: float
    switching_count: int
    mean_sinr_db: float
    slot_runtime_ms: float = 0.0


@dataclass
class EpisodeSummary:
    """Aggregated summary statistics over an evaluation episode."""
    total_served_mb: float = 0.0
    mean_throughput_mbps: float = 0.0
    cumulative_reward: float = 0.0
    mean_reward_per_slot: float = 0.0
    initial_backlog_mb: float = 0.0
    final_backlog_mb: float = 0.0
    net_queue_growth_mb: float = 0.0
    mean_retention_rate: float = 0.0
    mean_conflict_rate: float = 0.0
    mean_interference_watts: float = 0.0
    mean_jains_fairness: float = 0.0
    mean_rb_utilization: float = 0.0
    mean_power_utilization: float = 0.0
    total_beam_switches: int = 0
    mean_sinr_db: float = 0.0
    mean_runtime_ms: float = 0.0
    slot_metrics: List[SlotMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """Convert scalar summary metrics to dictionary."""
        return {
            "total_served_mb": float(self.total_served_mb),
            "mean_throughput_mbps": float(self.mean_throughput_mbps),
            "cumulative_reward": float(self.cumulative_reward),
            "mean_reward_per_slot": float(self.mean_reward_per_slot),
            "final_backlog_mb": float(self.final_backlog_mb),
            "net_queue_growth_mb": float(self.net_queue_growth_mb),
            "mean_retention_rate": float(self.mean_retention_rate),
            "mean_conflict_rate": float(self.mean_conflict_rate),
            "mean_interference_watts": float(self.mean_interference_watts),
            "mean_jains_fairness": float(self.mean_jains_fairness),
            "mean_rb_utilization": float(self.mean_rb_utilization),
            "mean_power_utilization": float(self.mean_power_utilization),
            "total_beam_switches": int(self.total_beam_switches),
            "mean_sinr_db": float(self.mean_sinr_db),
            "mean_runtime_ms": float(self.mean_runtime_ms),
        }


class MetricsTracker:
    """Calculates and aggregates network-level performance metrics."""

    @staticmethod
    def compute_jains_fairness(allocations: np.ndarray) -> float:
        """
        Jain's Fairness Index:
            J(x) = (sum(x_i))^2 / (N * sum(x_i^2))
        """
        n = len(allocations)
        if n == 0:
            return 1.0
        s = float(np.sum(allocations))
        sq = float(np.sum(allocations**2))
        if sq <= 0.0:
            return 1.0
        return float((s**2) / (n * sq))

    @staticmethod
    def aggregate_episode(slot_records: List[SlotMetrics], initial_backlog: float) -> EpisodeSummary:
        """
        Aggregate a sequence of slot metrics into an EpisodeSummary.
        """
        if not slot_records:
            return EpisodeSummary()

        summary = EpisodeSummary()
        summary.slot_metrics = slot_records
        summary.initial_backlog_mb = initial_backlog

        served_list = [m.served_traffic_mb for m in slot_records]
        throughput_list = [m.throughput_mbps for m in slot_records]
        reward_list = [m.team_reward for m in slot_records]
        retention_list = [m.retention_rate for m in slot_records]
        conflict_list = [m.conflict_rate for m in slot_records]
        interf_list = [m.interference_watts for m in slot_records]
        fairness_list = [m.jains_fairness for m in slot_records]
        rb_util_list = [m.rb_utilization for m in slot_records]
        power_util_list = [m.power_utilization for m in slot_records]
        sinr_list = [m.mean_sinr_db for m in slot_records]
        runtime_list = [m.slot_runtime_ms for m in slot_records]

        summary.total_served_mb = float(np.sum(served_list))
        summary.mean_throughput_mbps = float(np.mean(throughput_list))
        summary.cumulative_reward = float(np.sum(reward_list))
        summary.mean_reward_per_slot = float(np.mean(reward_list))
        summary.final_backlog_mb = float(slot_records[-1].total_backlog_mb)
        summary.net_queue_growth_mb = float(summary.final_backlog_mb - summary.initial_backlog_mb)
        summary.mean_retention_rate = float(np.mean(retention_list))
        summary.mean_conflict_rate = float(np.mean(conflict_list))
        summary.mean_interference_watts = float(np.mean(interf_list))
        summary.mean_jains_fairness = float(np.mean(fairness_list))
        summary.mean_rb_utilization = float(np.mean(rb_util_list))
        summary.mean_power_utilization = float(np.mean(power_util_list))
        summary.total_beam_switches = int(sum(m.switching_count for m in slot_records))
        summary.mean_sinr_db = float(np.mean(sinr_list))
        summary.mean_runtime_ms = float(np.mean(runtime_list))

        return summary

