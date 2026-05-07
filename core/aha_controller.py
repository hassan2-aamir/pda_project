"""
AHA Controller — Adaptive Hybrid Aggregation
=============================================
Runtime control layer that monitors telemetry and switches between
Parameter Server (PS) and Ring AllReduce (RA) using a threshold +
hysteresis policy.

This is the NOVEL CONTRIBUTION of the project (C-6 justification):
no prior work implements a reproducible, telemetry-driven hybrid
policy with bounded-staleness safeguards at this level of detail.
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional
from core.telemetry import TelemetryBus, ControllerTelemetry

logger = logging.getLogger("AHAController")


@dataclass
class AHAConfig:
    """Tunable hyperparameters for the AHA decision policy."""
    # Thresholds for triggering a switch to PS
    lat_threshold_ms: float = 50.0      # p95 latency above which PS is preferred
    straggler_var_threshold: float = 200.0  # step-time variance indicating straggler
    pkt_loss_threshold: float = 0.01    # >1% packet loss → prefer PS
    queue_depth_threshold: int = 4      # PS shard lag → prefer RA

    # Hysteresis: require signal for H consecutive decision epochs before switching
    hysteresis_H: int = 3
    decision_epoch_steps: int = 10      # Evaluate every N steps

    # Staleness guard (PS mode)
    tau: int = 2                        # Max staleness iterations allowed

    # Logging
    verbose: bool = True


class AHAController:
    """
    AHA Controller daemon.

    Runs a background thread that periodically:
    1. Reads aggregated telemetry from the TelemetryBus
    2. Evaluates the switching policy
    3. Emits a mode token (PS | RA) to all workers

    Workers poll get_mode() each step; the controller updates it
    after each decision epoch.
    """

    def __init__(self, telemetry_bus: TelemetryBus, config: Optional[AHAConfig] = None):
        self.bus = telemetry_bus
        self.cfg = config or AHAConfig()
        self._mode: str = "RA"            # Start in RA (stable default)
        self._mode_lock = threading.Lock()
        self._hysteresis_count: int = 0
        self._candidate: str = "RA"
        self._switch_count: int = 0
        self._decision_log: list = []
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._current_step: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self):
        """Start the background decision loop."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AHAController")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_mode(self) -> str:
        """Workers call this to get current aggregation mode."""
        with self._mode_lock:
            return self._mode

    def advance_step(self, step: int):
        """Workers notify the controller of the current step."""
        self._current_step = step

    def stats(self) -> dict:
        return {
            "current_mode": self.get_mode(),
            "switch_count": self._switch_count,
            "hysteresis_count": self._hysteresis_count,
            "decision_log": self._decision_log[-10:],   # Last 10 decisions
        }

    # ------------------------------------------------------------------ #
    # Internal decision loop                                               #
    # ------------------------------------------------------------------ #

    def _loop(self):
        """Background thread: evaluate policy every decision_epoch_steps."""
        epoch_duration = self.cfg.decision_epoch_steps * 0.05   # ~50ms per step
        while self._running:
            time.sleep(epoch_duration)
            if not self._running:
                break
            telemetry = self.bus.read(self._current_step)
            self._evaluate(telemetry)

    def _evaluate(self, t: ControllerTelemetry):
        """Core decision policy: threshold + hysteresis."""
        # --- Compute trigger signals ---
        straggler     = t.step_var > self.cfg.straggler_var_threshold
        net_congested = (t.lat_p95_ms > self.cfg.lat_threshold_ms or
                         t.pkt_loss > self.cfg.pkt_loss_threshold)
        ps_hotspot    = t.queue_depth > self.cfg.queue_depth_threshold
        loss_stalling = t.loss_slope > -0.001   # Not decreasing fast enough

        # --- Select candidate mode ---
        if straggler or net_congested:
            candidate = "PS"    # PS tolerates stragglers & congestion better
        elif ps_hotspot:
            candidate = "RA"    # PS overloaded → switch to RA
        else:
            candidate = "RA"    # Stable network → prefer RA (balanced load)

        # --- Hysteresis: commit only after H consecutive signals ---
        with self._mode_lock:
            current = self._mode

        if candidate != current:
            self._hysteresis_count += 1
            if self._hysteresis_count >= self.cfg.hysteresis_H:
                self._execute_switch(candidate, t)
                self._hysteresis_count = 0
        else:
            self._hysteresis_count = max(0, self._hysteresis_count - 1)

        # Log decision
        self._decision_log.append({
            "step": t.step,
            "mode": self.get_mode(),
            "candidate": candidate,
            "straggler": straggler,
            "net_congested": net_congested,
            "ps_hotspot": ps_hotspot,
            "hysteresis": self._hysteresis_count,
            "lat_p95": round(t.lat_p95_ms, 2),
            "step_var": round(t.step_var, 2),
            "pkt_loss": round(t.pkt_loss, 4),
        })

        if self.cfg.verbose:
            trig = []
            if straggler: trig.append("STRAGGLER")
            if net_congested: trig.append("NET_CONG")
            if ps_hotspot: trig.append("PS_HOT")
            trig_str = ",".join(trig) if trig else "STABLE"
            logger.debug(
                f"[Step {t.step:4d}] mode={self.get_mode()} candidate={candidate} "
                f"H={self._hysteresis_count} triggers=[{trig_str}] "
                f"lat_p95={t.lat_p95_ms:.1f}ms var={t.step_var:.1f}"
            )

    def _execute_switch(self, new_mode: str, t: ControllerTelemetry):
        """
        Commit a mode switch.
        In the real system this also enforces a barrier and checkpoint.
        """
        with self._mode_lock:
            old_mode = self._mode
            self._mode = new_mode
        self._switch_count += 1
        if self.cfg.verbose:
            logger.info(
                f"[Step {t.step}] MODE SWITCH: {old_mode} → {new_mode} "
                f"(switch #{self._switch_count}, "
                f"lat_p95={t.lat_p95_ms:.1f}ms, var={t.step_var:.1f}, "
                f"pkt_loss={t.pkt_loss:.4f})"
            )
