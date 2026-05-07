"""
AHA Telemetry Bus
-----------------
Collects per-step metrics from each worker and makes them available
to the AHA controller for mode-switch decisions.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class StepTelemetry:
    """Single step telemetry record emitted by a worker."""
    worker_id: int
    step: int
    step_time_ms: float          # Wall-clock step duration
    grad_bytes: int              # Gradient tensor size in bytes
    comm_time_ms: float          # Communication-only time
    pkt_loss_est: float          # Estimated packet loss [0, 1]
    mode: str                    # 'PS' or 'RA'
    timestamp: float = field(default_factory=time.time)


@dataclass
class ControllerTelemetry:
    """Aggregated telemetry consumed by the AHA controller."""
    step: int
    lat_p95_ms: float            # 95th percentile step latency across workers
    step_var: float              # Variance in per-worker step times
    queue_depth: int             # PS queue depth proxy (max worker lag)
    pkt_loss: float              # Average packet loss estimate
    grad_bytes: int              # Gradient size
    loss_slope: float            # Training loss slope (recent W steps)
    n_workers: int


class TelemetryBus:
    """
    Shared-memory telemetry bus (simulated via thread-safe queues).
    Workers push StepTelemetry; the controller reads ControllerTelemetry.
    """

    def __init__(self, n_workers: int, window: int = 20):
        self.n_workers = n_workers
        self.window = window                         # Rolling window for stats
        self._lock = threading.Lock()
        self._worker_records: Dict[int, deque] = {
            i: deque(maxlen=window) for i in range(n_workers)
        }
        self._loss_history: deque = deque(maxlen=50)
        self._step_records: Dict[int, List[StepTelemetry]] = {}

    # ------------------------------------------------------------------ #
    # Worker-side push                                                     #
    # ------------------------------------------------------------------ #

    def push(self, record: StepTelemetry):
        with self._lock:
            self._worker_records[record.worker_id].append(record)
            step = record.step
            if step not in self._step_records:
                self._step_records[step] = []
            self._step_records[step].append(record)

    def push_loss(self, loss: float):
        with self._lock:
            self._loss_history.append(loss)

    # ------------------------------------------------------------------ #
    # Controller-side read                                                 #
    # ------------------------------------------------------------------ #

    def read(self, current_step: int) -> ControllerTelemetry:
        with self._lock:
            all_records = []
            for q in self._worker_records.values():
                all_records.extend(list(q))

            if not all_records:
                return ControllerTelemetry(
                    step=current_step, lat_p95_ms=0.0, step_var=0.0,
                    queue_depth=0, pkt_loss=0.0, grad_bytes=0,
                    loss_slope=0.0, n_workers=self.n_workers
                )

            step_times = [r.step_time_ms for r in all_records]
            # Per-worker latest step to calculate lag (queue depth proxy)
            latest_steps = {}
            for r in all_records:
                if r.worker_id not in latest_steps or r.step > latest_steps[r.worker_id]:
                    latest_steps[r.worker_id] = r.step
            step_values = list(latest_steps.values())
            queue_depth = max(step_values) - min(step_values) if step_values else 0

            pkt_losses = [r.pkt_loss_est for r in all_records]
            grad_bytes = all_records[-1].grad_bytes if all_records else 0

            # Loss slope over recent history
            loss_hist = list(self._loss_history)
            if len(loss_hist) >= 5:
                x = np.arange(len(loss_hist))
                slope = float(np.polyfit(x, loss_hist, 1)[0])
            else:
                slope = -1.0   # Decreasing by default (normal early training)

            return ControllerTelemetry(
                step=current_step,
                lat_p95_ms=float(np.percentile(step_times, 95)) if step_times else 0.0,
                step_var=float(np.var(step_times)) if len(step_times) > 1 else 0.0,
                queue_depth=queue_depth,
                pkt_loss=float(np.mean(pkt_losses)) if pkt_losses else 0.0,
                grad_bytes=grad_bytes,
                loss_slope=slope,
                n_workers=self.n_workers,
            )


class MetricsLogger:
    """Records experiment metrics for post-run analysis and plotting."""

    def __init__(self, label: str):
        self.label = label
        self.steps: List[int] = []
        self.throughput: List[float] = []   # samples/sec
        self.step_time_ms: List[float] = [] # median step time
        self.p95_step_time: List[float] = []
        self.comm_time_ms: List[float] = []
        self.loss: List[float] = []
        self.accuracy: List[float] = []
        self.mode_history: List[str] = []   # 'PS' or 'RA'
        self.switch_events: List[int] = []  # Steps where mode switched
        self.fault_events: List[int] = []
        self._lock = threading.Lock()

    def log_step(self, step: int, step_times: List[float],
                 comm_times: List[float], loss: float, accuracy: float,
                 mode: str, batch_size: int):
        with self._lock:
            self.steps.append(step)
            median_step = float(np.median(step_times)) if step_times else 0.0
            self.step_time_ms.append(median_step)
            self.p95_step_time.append(float(np.percentile(step_times, 95)) if step_times else 0.0)
            self.comm_time_ms.append(float(np.mean(comm_times)) if comm_times else 0.0)
            # Throughput: (n_workers * batch_size) / step_time_seconds
            n = len(step_times) if step_times else 1
            step_sec = median_step / 1000.0 if median_step > 0 else 1e-3
            self.throughput.append((n * batch_size) / step_sec)
            self.loss.append(loss)
            self.accuracy.append(accuracy)
            self.mode_history.append(mode)

    def log_switch(self, step: int):
        with self._lock:
            self.switch_events.append(step)

    def log_fault(self, step: int):
        with self._lock:
            self.fault_events.append(step)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "final_accuracy": self.accuracy[-1] if self.accuracy else 0.0,
            "mean_throughput": float(np.mean(self.throughput)) if self.throughput else 0.0,
            "mean_step_ms": float(np.mean(self.step_time_ms)) if self.step_time_ms else 0.0,
            "p95_step_ms": float(np.mean(self.p95_step_time)) if self.p95_step_time else 0.0,
            "n_switches": len(self.switch_events),
            "total_steps": len(self.steps),
        }
