Explanation of core modules (line-by-line)
=========================================

This document walks through each module in the `core/` package and explains its contents. Explanations are given in-order (approximating a line-by-line walkthrough) for readability — grouped where consecutive lines form a single logical unit.

**core/__init__.py**
- File is intentionally empty. It simply marks `core/` as a Python package.

**core/aha_controller.py**
- Module docstring: high-level description of the AHA (Adaptive Hybrid Aggregation) controller and its purpose.
- Imports: `time`, `threading`, `logging`, `dataclasses.dataclass`, `typing.Optional`, and types from `core.telemetry`.
- `logger = logging.getLogger("AHAController")`: module-level logger for controller messages.

- `@dataclass class AHAConfig:`: container for tunable hyperparameters. Fields explained:
  - `lat_threshold_ms`: p95 latency above which PS is preferred.
  - `straggler_var_threshold`: variance threshold indicating stragglers.
  - `pkt_loss_threshold`: packet-loss fraction threshold.
  - `queue_depth_threshold`: queue-depth threshold used as a PS hotspot proxy.
  - `hysteresis_H`: number of consecutive epochs required to commit a switch.
  - `decision_epoch_steps`: number of worker steps per controller epoch.
  - `tau`: staleness guard for PS mode (max allowed staleness iterations).
  - `verbose`: toggle for extra logging.

- `class AHAController:`: the controller daemon. Key attributes created in `__init__`:
  - `bus`: TelemetryBus instance used to read aggregated telemetry.
  - `cfg`: configuration (AHAConfig).
  - `_mode`, `_mode_lock`: current aggregation mode and lock for thread-safe reads/writes.
  - `_hysteresis_count`: counter tracking consecutive candidate signals.
  - `_candidate`, `_switch_count`, `_decision_log`: bookkeeping for decisions.
  - `_running`, `_thread`: background thread control.
  - `_current_step`: last step seen from workers.

- `start()` / `stop()`: start/stop the background decision loop thread. `start` spawns a daemon thread named "AHAController".

- `get_mode()`: thread-safe getter for the current mode (workers call this each step).
- `advance_step(step)`: controller receives current worker step (used for aligning telemetry reads).
- `stats()`: returns a small dict of controller state (current_mode, switch_count, hysteresis_count, recent decision_log).

- `_loop()`: background loop executed by the thread. Sleeps `epoch_duration` (derived from `decision_epoch_steps`) and then reads telemetry for `_current_step` and calls `_evaluate()`.

- `_evaluate(t: ControllerTelemetry)`: core decision routine (threshold + hysteresis):
  - Compute trigger booleans from telemetry: `straggler`, `net_congested`, `ps_hotspot`, and `loss_stalling`.
  - Choose `candidate` mode based on triggers: PS if straggler or network congestion; RA if PS is a hotspot or default stable.
  - Apply hysteresis: only execute switch if candidate differs from current mode for `hysteresis_H` consecutive epochs; otherwise decrement hysteresis counter when candidate == current.
  - Append a compact decision entry to `_decision_log` including numeric metrics.
  - If verbose, log a debug line summarizing the decision and triggers.

- `_execute_switch(new_mode, t)`: commit a mode switch by swapping `_mode` under lock, increment `_switch_count`, and log info when verbose. In a real system this is where barriers/checkpoints would be enforced.

Notes: The controller is intentionally conservative (hysteresis + staleness guard) to avoid oscillation. Workers poll `get_mode()` each step to adapt behavior.

**core/backends.py**
- Module docstring: simulation of PS and Ring AllReduce backends using Python primitives and artificial time delays to model network costs.
- Imports: `time`, `threading`, `numpy as np`, typing helpers, and `dataclass`.

- `class NetworkLink:`
  - Models bandwidth, base latency, packet loss, jitter.
  - `simulate_transfer(n_bytes)`: computes transmission time = base latency + (n_bytes*8 / bandwidth_bps) + jitter, includes retransmit modeling when packet loss > 0, sleeps for that duration, and returns the transfer time in ms.
  - `set_congestion(factor)`: scales `base_latency_ms` and `jitter_ms` (simple congestion model).
  - `inject_loss(rate)`: set packet loss rate.

- `@dataclass class GradientTensor:`
  - Lightweight simulated gradient container with `values` (numpy array), `version`, and `worker_id`.
  - `n_bytes` property returns `values.nbytes`.
  - `random(size_mb, worker_id, version, seed)`: classmethod to synthesize a random gradient array of roughly `size_mb` megabytes (float32).

- `class ParameterServer:` (sharded PS with τ-BSP semantics)
  - __init__: sets `n_shards`, `tau`, `link`, internal locks, shards dict, versions per shard, per-worker versions, `_global_version`, `_staleness_violations`, `_total_pushes`, and `_queue_depth` proxy.
  - `_shard_for(grad)`: partitions a gradient array into `n_shards` contiguous slices.
  - `push(grad)`: simulates transfer for each shard slice (calls `link.simulate_transfer`), then under lock checks staleness (`lag = global_version - grad.version`), rejects stale updates if `lag > tau` (counts staleness violations). Otherwise accumulates shard slices (simple averaging), increments versions and `_global_version`, updates worker's seen version, and returns communication time in ms.
  - `pull(grad_size_bytes)`: simulates parameter pull transfer time (one slice per shard), returns concatenated shards or zeros if none.
  - `stats()`: returns PS metrics like `global_version`, `staleness_violations`, `total_pushes`, `violation_rate`, and `queue_depth`.
  - `reset()`: clears shards and resets counters.

- `class RingAllReduce:` (synchronous ring allreduce simulation)
  - __init__: takes `worker_ids` and `link`, creates a `threading.Barrier` sized to number of workers, `_chunks` aggregated storage and per-worker readiness flags, `_n_workers`, and `_ring_rebuild_count`.
  - `remove_worker(worker_id)`: remove a failed worker, rebuild barrier with the new size, increment `ring_rebuild_count`.
  - `allreduce(grad, worker_id)`: simulates scatter-reduce and all-gather phases by performing `N-1` simulated transfers for each phase (total `2*(N-1)` hops of `chunk_bytes`), stores the averaged chunk under lock, waits on barrier (with timeout); on barrier error returns partial result. After barrier, computes the reduced result (mean of `_chunks`) and returns it plus total comm time.
  - `stats()`: returns `n_workers` and `ring_rebuild_count`.

Design notes: Both backends use `NetworkLink.simulate_transfer` to make communication costs explicit and stochastic (jitter + retransmit), enabling the controller and workers to observe latencies and packet loss.

**core/telemetry.py**
- Module docstring: telemetry bus for collecting per-step metrics and exposing aggregated ControllerTelemetry to the AHA controller.
- Imports: `time`, `threading`, `collections.deque`, `dataclass/field`, `typing` helpers, and `numpy as np`.

- `@dataclass StepTelemetry`: record emitted by a worker each step. Fields:
  - `worker_id`, `step`, `step_time_ms`, `grad_bytes`, `comm_time_ms`, `pkt_loss_est`, `mode`, and `timestamp`.

- `@dataclass ControllerTelemetry`: aggregated view consumed by controller. Fields include `step`, `lat_p95_ms`, `step_var`, `queue_depth`, `pkt_loss`, `grad_bytes`, `loss_slope`, and `n_workers`.

- `class TelemetryBus:`
  - __init__(n_workers, window): creates per-worker deques (maxlen=window), `_loss_history`, and `_step_records` dict.
  - `push(record)`: append a StepTelemetry record to that worker's deque and add to `_step_records` keyed by step.
  - `push_loss(loss)`: append loss to `_loss_history` for slope computation.
  - `read(current_step)`: aggregate recent records across workers, compute p95 step latency, variance, queue_depth proxy (max - min of latest per-worker steps), mean packet loss estimate, latest `grad_bytes`, and `loss_slope` by fitting a linear trend to `_loss_history` (if >=5 samples). Returns a ControllerTelemetry instance. If no records, returns zeros/defaults.

- `class MetricsLogger:`
  - Simple thread-safe collector for experiment metrics: steps, throughput, median/p95 step times, comm times, loss, accuracy, mode history, switch & fault events.
  - `log_step(...)`: append computed per-step aggregates (median/p95 step times, mean comm, throughput = (n_workers * batch_size) / step_seconds, loss, accuracy, mode).
  - `log_switch(step)` / `log_fault(step)`: record discrete events.
  - `summary()`: return a small dict summary with `final_accuracy`, `mean_throughput`, `mean_step_ms`, `p95_step_ms`, `n_switches`, and `total_steps`.

**core/worker.py**
- Module docstring: simulated worker training loop; integrates model compute, backend comms, telemetry push, and optional fault injection.
- Imports: `time`, `threading`, `numpy as np`, `logging`, dataclasses, typing, and references to `core.telemetry`, `core.backends`, and `core.aha_controller`.

- `class SimulatedModel:`
  - Lightweight simulation of compute + loss/accuracy dynamics.
  - `grad_size_mb`, `compute_ms` and RNG seeded by `seed`.
  - `forward_backward(step, worker_id, compute_noise_ms)`: simulate compute by sleeping for `actual_compute` ms (variability via RNG), then return `GradientTensor.random(...)` seeded with RNG value.
  - `update(reduced_grad, lr)`: simulate updating internal loss & accuracy with small stochastic changes.
  - `loss` and `accuracy` properties expose current simulated metrics.

- `@dataclass FaultConfig`: schedule for injecting faults and network anomalies (straggler steps, slowdown ms, crash step, bandwidth throttle steps and factor, packet loss steps and rate). `__post_init__` normalizes None lists to empty lists.

- `class Worker:`
  - Orchestrates a per-worker training loop and exposes `run_steps(n_steps, metrics, on_step_complete)`.
  - Key attributes: ids, model, ps, ring, telemetry bus, link, mode (`'PS'`, `'RA'`, or `'AHA'`), controller (optional), fault config, batch size, alive flag, per-step timing arrays, and current step.
  - `run_steps`: for each step:
    - Early check for alive; advance controller step if present.
    - Apply fault injections (straggler slowdowns, crash, bandwidth throttle, packet loss). Updates link and logs events. On crash, mark not alive, remove worker from ring and log fault event.
    - Call `model.forward_backward(...)` to produce a `GradientTensor`.
    - Determine `current_mode` via `_get_mode()` (queries controller if `AHA`).
    - If `PS`: call `ps.push(grad)` then `ps.pull(...)`, aggregate comm time; if `RA`: call `ring.allreduce(grad, self.worker_id)`.
    - Call `model.update(reduced)`.
    - Compute `step_ms` (wall-clock) and push a `StepTelemetry` record to `TelemetryBus` with step_time_ms, grad_bytes, comm_time_ms, pkt_loss_est, and mode.
    - Append local timing lists and optionally call `on_step_complete` callback.

- `_get_mode()`: helper to return effective mode; when `fixed_mode == 'AHA'` it queries `controller.get_mode()` falling back to `'RA'` if no controller is present.

- `alive`, `step_times`, `comm_times`: simple property accessors.

Usage notes
-----------
- The codebase is a simulation useful for experiments: the `NetworkLink` makes comm costs explicit and stochastic; `ParameterServer` and `RingAllReduce` implement different aggregation semantics; `TelemetryBus` aggregates worker step metrics; `AHAController` reads telemetry and issues mode decisions; `Worker` integrates all pieces and produces telemetry.

If you want a literal annotated copy of each file (each source line followed by an inline explanation), I can produce a fully annotated version of a single file first (e.g., `core/worker.py`) to confirm the format you prefer, then expand to every file. Which file should I annotate first in that precise, per-line inline style?
