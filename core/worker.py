"""
Worker Node Simulation
======================
Each worker simulates the training loop:
  forward pass → backward pass → gradient emission → aggregation → parameter update

Faults can be injected via the FaultInjector.
"""

import time
import threading
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Callable, List

from core.telemetry import TelemetryBus, StepTelemetry, MetricsLogger
from core.backends import ParameterServer, RingAllReduce, GradientTensor, NetworkLink
from core.aha_controller import AHAController

logger = logging.getLogger("Worker")


# ════════════════════════════════════════════════════════════════════════ #
# Simulated training model                                                 #
# ════════════════════════════════════════════════════════════════════════ #

class SimulatedModel:
    """
    Lightweight training simulation.
    The model has no real neural layers — it simulates:
    - Compute time (forward + backward) scaled by compute_ms
    - Loss that decreases stochastically over steps
    - Accuracy that increases accordingly
    Gradient size is fixed at grad_size_mb megabytes.
    """

    def __init__(self, grad_size_mb: float = 10.0, compute_ms: float = 20.0,
                 seed: int = 42):
        self.grad_size_mb = grad_size_mb
        self.compute_ms = compute_ms
        self.rng = np.random.default_rng(seed)
        self._loss = 2.5           # Starting loss
        self._accuracy = 0.10      # Starting accuracy

    def forward_backward(self, step: int, worker_id: int,
                         compute_noise_ms: float = 0.0) -> GradientTensor:
        """Simulate forward + backward pass. Returns gradient tensor."""
        # Add compute noise (simulates hardware variance / straggler)
        actual_compute = max(1.0, self.compute_ms + compute_noise_ms +
                             self.rng.normal(0, self.compute_ms * 0.05))
        time.sleep(actual_compute / 1000.0)
        return GradientTensor.random(self.grad_size_mb, worker_id, step,
                                     seed=self.rng.integers(1000))

    def update(self, reduced_grad: np.ndarray, lr: float = 0.01):
        """Apply gradient update. Updates loss and accuracy."""
        # Simulate loss decrease with noise
        noise = self.rng.normal(0, 0.05)
        self._loss = max(0.05, self._loss * 0.985 + noise * 0.1)
        self._accuracy = min(0.99, 1.0 - self._loss / 3.0 + self.rng.uniform(0, 0.02))

    @property
    def loss(self): return self._loss
    @property
    def accuracy(self): return self._accuracy


# ════════════════════════════════════════════════════════════════════════ #
# Fault Injector                                                           #
# ════════════════════════════════════════════════════════════════════════ #

@dataclass
class FaultConfig:
    """Fault injection schedule."""
    straggler_steps: List[int] = None          # Steps where this worker is slow
    straggler_slowdown_ms: float = 200.0       # Extra compute delay in ms
    crash_at_step: Optional[int] = None        # Step where worker crashes
    bandwidth_throttle_steps: List[int] = None # Steps where link is throttled
    throttle_factor: float = 0.1              # 10% of normal bandwidth
    pkt_loss_steps: List[int] = None
    pkt_loss_rate: float = 0.05

    def __post_init__(self):
        self.straggler_steps = self.straggler_steps or []
        self.bandwidth_throttle_steps = self.bandwidth_throttle_steps or []
        self.pkt_loss_steps = self.pkt_loss_steps or []


# ════════════════════════════════════════════════════════════════════════ #
# Worker                                                                   #
# ════════════════════════════════════════════════════════════════════════ #

class Worker:
    """
    Distributed training worker.

    Supports three modes:
    - 'PS'  : push gradients to PS, pull updated params
    - 'RA'  : participate in ring allreduce barrier
    - 'AHA' : use mode token from AHAController to choose PS or RA
    """

    def __init__(self, worker_id: int, model: SimulatedModel,
                 ps: ParameterServer, ring: RingAllReduce,
                 telemetry_bus: TelemetryBus,
                 link: NetworkLink,
                 mode: str = 'AHA',          # 'PS', 'RA', or 'AHA'
                 controller: Optional[AHAController] = None,
                 fault_config: Optional[FaultConfig] = None,
                 batch_size: int = 32):
        self.worker_id = worker_id
        self.model = model
        self.ps = ps
        self.ring = ring
        self.bus = telemetry_bus
        self.link = link
        self.fixed_mode = mode
        self.controller = controller
        self.fault_cfg = fault_config or FaultConfig()
        self.batch_size = batch_size
        self._alive = True
        self._thread: Optional[threading.Thread] = None
        self._step_times: List[float] = []
        self._comm_times: List[float] = []
        self._current_step = 0

    def run_steps(self, n_steps: int, metrics: MetricsLogger,
                  on_step_complete: Optional[Callable] = None):
        """Run n_steps of the training loop. Blocks until done or crash."""
        for step in range(n_steps):
            if not self._alive:
                break

            step_start = time.perf_counter()
            self._current_step = step
            if self.controller:
                self.controller.advance_step(step)

            # ---- Fault injection ----
            compute_noise = 0.0
            pkt_loss = 0.0

            if step in self.fault_cfg.straggler_steps:
                compute_noise = self.fault_cfg.straggler_slowdown_ms
                logger.debug(f"W{self.worker_id} STRAGGLER at step {step}")

            if step == self.fault_cfg.crash_at_step:
                logger.info(f"W{self.worker_id} CRASH at step {step}")
                self._alive = False
                self.ring.remove_worker(self.worker_id)
                metrics.log_fault(step)
                break

            if step in self.fault_cfg.bandwidth_throttle_steps:
                self.link.bandwidth_gbps *= self.fault_cfg.throttle_factor
                logger.debug(f"W{self.worker_id} BW THROTTLE at step {step}")

            if step in self.fault_cfg.pkt_loss_steps:
                pkt_loss = self.fault_cfg.pkt_loss_rate
                self.link.inject_loss(pkt_loss)
            else:
                self.link.inject_loss(0.0)

            # ---- Compute: forward + backward ----
            grad = self.model.forward_backward(step, self.worker_id, compute_noise)

            # ---- Communication ----
            current_mode = self._get_mode()
            comm_time_ms = 0.0

            if current_mode == 'PS':
                comm_time_ms += self.ps.push(grad)
                _, pull_time = self.ps.pull(grad.n_bytes)
                comm_time_ms += pull_time
                reduced = grad.values  # Use local gradient (PS applies server-side avg)

            else:  # RA
                reduced, comm_time_ms = self.ring.allreduce(grad, self.worker_id)

            # ---- Model update ----
            self.model.update(reduced)

            # ---- Timing ----
            step_end = time.perf_counter()
            step_ms = (step_end - step_start) * 1000.0

            # ---- Telemetry push ----
            self.bus.push(StepTelemetry(
                worker_id=self.worker_id,
                step=step,
                step_time_ms=step_ms,
                grad_bytes=grad.n_bytes,
                comm_time_ms=comm_time_ms,
                pkt_loss_est=pkt_loss,
                mode=current_mode,
            ))

            self._step_times.append(step_ms)
            self._comm_times.append(comm_time_ms)

            if on_step_complete:
                on_step_complete(step, step_ms, comm_time_ms, current_mode)

    def _get_mode(self) -> str:
        if self.fixed_mode == 'AHA':
            return self.controller.get_mode() if self.controller else 'RA'
        return self.fixed_mode

    @property
    def alive(self): return self._alive
    @property
    def step_times(self): return list(self._step_times)
    @property
    def comm_times(self): return list(self._comm_times)
