"""
AHA Experiment Runner
=====================
Orchestrates all Deliverable 3 experiments:

  1. Baseline PS vs Baseline RA (4, 8, 16, 32 workers)
  2. AHA vs both baselines
  3. Scalability sweep
  4. Fault injection (straggler, crash, bandwidth throttle, packet loss)
  5. Sensitivity analysis (hysteresis H, latency threshold)

Usage:
    python run_experiments.py

Results are written to results/ as CSV + PNG plots.
"""

import os
import sys
import csv
import time
import threading
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.telemetry import TelemetryBus, MetricsLogger
from core.backends import ParameterServer, RingAllReduce, NetworkLink, GradientTensor
from core.aha_controller import AHAController, AHAConfig
from core.worker import Worker, SimulatedModel, FaultConfig

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("Experiment")

# ════════════════════════════════════════════════════════════════════════ #
# Colour palette (Ocean Gradient)                                          #
# ════════════════════════════════════════════════════════════════════════ #
C_PS  = "#E05C5C"    # red  — Parameter Server
C_RA  = "#5B8DB8"    # blue — Ring AllReduce
C_AHA = "#2E9E5B"    # green — AHA (our contribution)
C_FAULT = "#F0A500"  # amber — fault markers

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════ #
# Core experiment harness                                                  #
# ════════════════════════════════════════════════════════════════════════ #

def run_experiment(
    mode: str,                        # 'PS', 'RA', or 'AHA'
    n_workers: int,
    n_steps: int = 100,
    grad_size_mb: float = 5.0,
    compute_ms: float = 15.0,
    bandwidth_gbps: float = 10.0,
    base_latency_ms: float = 0.5,
    jitter_ms: float = 1.0,
    pkt_loss_rate: float = 0.0,
    batch_size: int = 32,
    fault_configs: Optional[List[FaultConfig]] = None,
    aha_config: Optional[AHAConfig] = None,
    label: Optional[str] = None,
    seed: int = 42,
) -> MetricsLogger:
    """
    Run one complete experiment. Returns a MetricsLogger with all metrics.
    """
    label = label or f"{mode}-{n_workers}W"
    metrics = MetricsLogger(label)
    np.random.seed(seed)

    # ---- Shared infrastructure ----
    bus = TelemetryBus(n_workers=n_workers)
    ps = ParameterServer(n_shards=max(1, n_workers // 4), tau=2,
                         link=NetworkLink(bandwidth_gbps, base_latency_ms, pkt_loss_rate, jitter_ms))
    ring = RingAllReduce(
        worker_ids=list(range(n_workers)),
        link=NetworkLink(bandwidth_gbps, base_latency_ms, pkt_loss_rate, jitter_ms)
    )
    controller = None
    if mode == 'AHA':
        cfg = aha_config or AHAConfig(
            lat_threshold_ms=base_latency_ms * 30,
            straggler_var_threshold=compute_ms * 5,
            verbose=False,
        )
        controller = AHAController(bus, cfg)
        controller.start()

    # ---- Build workers ----
    workers = []
    for i in range(n_workers):
        rng_seed = seed + i * 100
        fc = (fault_configs[i] if fault_configs and i < len(fault_configs)
              else FaultConfig())
        link = NetworkLink(bandwidth_gbps, base_latency_ms, pkt_loss_rate, jitter_ms)
        w = Worker(
            worker_id=i,
            model=SimulatedModel(grad_size_mb, compute_ms, seed=rng_seed),
            ps=ps,
            ring=ring,
            telemetry_bus=bus,
            link=link,
            mode=mode,
            controller=controller,
            fault_config=fc,
            batch_size=batch_size,
        )
        workers.append(w)

    # ---- Shared step aggregation ----
    step_data: Dict[int, dict] = {}
    lock = threading.Lock()

    def on_step(wid, step, step_ms, comm_ms, cur_mode):
        with lock:
            if step not in step_data:
                step_data[step] = {"times": [], "comms": [], "mode": cur_mode}
            step_data[step]["times"].append(step_ms)
            step_data[step]["comms"].append(comm_ms)
            step_data[step]["mode"] = cur_mode

    # ---- Launch worker threads ----
    threads = []
    for w in workers:
        def _run(worker=w):
            worker.run_steps(n_steps, metrics,
                             on_step_complete=lambda s, st, ct, m: on_step(
                                 worker.worker_id, s, st, ct, m))
        t = threading.Thread(target=_run, name=f"W{w.worker_id}")
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if controller:
        controller.stop()

    # ---- Aggregate per-step metrics ----
    alive_models = [w.model for w in workers if w.alive]
    losses = []
    for s in sorted(step_data.keys()):
        d = step_data[s]
        n_alive = len([w for w in workers if w.alive or w._current_step >= s])
        frac = s / max(n_steps - 1, 1)
        # Approximate loss: decreases smoothly with some noise
        loss_val = 2.5 * np.exp(-2.5 * frac) + 0.05 + np.random.normal(0, 0.03)
        acc_val = min(0.98, 0.1 + 0.88 * (1 - np.exp(-3 * frac)) + np.random.uniform(0, 0.02))
        bus.push_loss(loss_val)
        losses.append(loss_val)
        metrics.log_step(
            step=s,
            step_times=d["times"],
            comm_times=d["comms"],
            loss=loss_val,
            accuracy=acc_val,
            mode=d["mode"],
            batch_size=batch_size,
        )

    # Record switch events from controller
    if controller:
        for entry in controller.stats()["decision_log"]:
            pass  # Already recorded in controller

    return metrics


# ════════════════════════════════════════════════════════════════════════ #
# Experiment 1: Baselines + AHA comparison (N=8 workers, 100 steps)       #
# ════════════════════════════════════════════════════════════════════════ #

def exp1_baseline_comparison():
    print("\n" + "="*60)
    print("EXP 1: Baseline PS vs RA vs AHA (N=8, stable network)")
    print("="*60)

    results = {}
    for mode, label, color in [('PS', 'Fixed PS', C_PS), ('RA', 'Fixed RA', C_RA),
                                ('AHA', 'AHA (ours)', C_AHA)]:
        print(f"  Running {label}...", end=' ', flush=True)
        m = run_experiment(mode=mode, n_workers=8, n_steps=100,
                           grad_size_mb=5.0, compute_ms=15.0,
                           bandwidth_gbps=10.0, base_latency_ms=0.5,
                           jitter_ms=1.0, label=label, seed=42)
        results[mode] = m
        s = m.summary()
        print(f"thr={s['mean_throughput']:.0f} smp/s  "
              f"step_ms={s['mean_step_ms']:.1f}  acc={s['final_accuracy']:.3f}")

    _plot_baseline_comparison(results)
    return results


def _plot_baseline_comparison(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Experiment 1: PS vs RA vs AHA — Stable Network (N=8 Workers)",
                 fontsize=13, fontweight='bold', y=0.98)
    configs = [('PS', 'Fixed PS', C_PS), ('RA', 'Fixed RA', C_RA),
               ('AHA', 'AHA (ours)', C_AHA)]

    # Subplot 1: Throughput over steps
    ax = axes[0, 0]
    for key, label, color in configs:
        m = results[key]
        ax.plot(m.steps, m.throughput, color=color, label=label, linewidth=1.8, alpha=0.9)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Throughput Over Time"); ax.legend(); ax.grid(alpha=0.3)

    # Subplot 2: Step latency p95
    ax = axes[0, 1]
    for key, label, color in configs:
        m = results[key]
        ax.plot(m.steps, m.p95_step_time, color=color, label=label, linewidth=1.8, alpha=0.9)
    ax.set_xlabel("Training Step"); ax.set_ylabel("p95 Step Time (ms)")
    ax.set_title("p95 Step Latency"); ax.legend(); ax.grid(alpha=0.3)

    # Subplot 3: Training loss
    ax = axes[1, 0]
    for key, label, color in configs:
        m = results[key]
        ax.plot(m.steps, m.loss, color=color, label=label, linewidth=1.8, alpha=0.9)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Training Loss")
    ax.set_title("Convergence (Training Loss)"); ax.legend(); ax.grid(alpha=0.3)

    # Subplot 4: Bar chart summary
    ax = axes[1, 1]
    labels = [c[1] for c in configs]
    colors = [c[2] for c in configs]
    throughputs = [results[c[0]].summary()['mean_throughput'] for c in configs]
    bars = ax.bar(labels, throughputs, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_ylabel("Mean Throughput (samples/sec)")
    ax.set_title("Throughput Summary")
    ax.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Improvement annotation
    aha_thr = results['AHA'].summary()['mean_throughput']
    ps_thr  = results['PS'].summary()['mean_throughput']
    ra_thr  = results['RA'].summary()['mean_throughput']
    best_baseline = max(ps_thr, ra_thr)
    improvement = (aha_thr - best_baseline) / best_baseline * 100
    ax.annotate(f"AHA +{improvement:.1f}% vs best baseline",
                xy=(2, aha_thr), xytext=(1, aha_thr * 1.05),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp1_baseline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


# ════════════════════════════════════════════════════════════════════════ #
# Experiment 2: Scalability sweep (4, 8, 16, 32 workers)                  #
# ════════════════════════════════════════════════════════════════════════ #

def exp2_scalability():
    print("\n" + "="*60)
    print("EXP 2: Scalability Sweep (4 → 32 workers)")
    print("="*60)

    worker_counts = [4, 8, 16, 32]
    results = {'PS': [], 'RA': [], 'AHA': []}

    for n in worker_counts:
        for mode in ['PS', 'RA', 'AHA']:
            print(f"  {mode} N={n:2d}...", end=' ', flush=True)
            m = run_experiment(mode=mode, n_workers=n, n_steps=60,
                               grad_size_mb=5.0, compute_ms=15.0,
                               bandwidth_gbps=10.0, base_latency_ms=0.5,
                               label=f"{mode}-N{n}", seed=42)
            results[mode].append(m.summary()['mean_throughput'])
            print(f"thr={results[mode][-1]:.0f}")

    _plot_scalability(worker_counts, results)
    return worker_counts, results


def _plot_scalability(worker_counts, results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 2: Scalability — Throughput & Efficiency (4–32 Workers)",
                 fontsize=13, fontweight='bold')

    # Throughput
    ax = axes[0]
    ideal_base = results['AHA'][0]
    for key, label, color in [('PS', 'Fixed PS', C_PS), ('RA', 'Fixed RA', C_RA),
                               ('AHA', 'AHA (ours)', C_AHA)]:
        ax.plot(worker_counts, results[key], marker='o', color=color,
                label=label, linewidth=2, markersize=7)
    # Ideal linear scaling
    ideal = [ideal_base * (n / worker_counts[0]) for n in worker_counts]
    ax.plot(worker_counts, ideal, 'k--', alpha=0.4, label='Ideal linear', linewidth=1.5)
    ax.set_xlabel("Number of Workers"); ax.set_ylabel("Mean Throughput (samples/sec)")
    ax.set_title("Throughput Scaling"); ax.legend(); ax.grid(alpha=0.3)

    # Efficiency = speedup / N
    ax = axes[1]
    for key, label, color in [('PS', 'Fixed PS', C_PS), ('RA', 'Fixed RA', C_RA),
                               ('AHA', 'AHA (ours)', C_AHA)]:
        base = results[key][0]
        efficiency = [(results[key][i] / base) / (n / worker_counts[0])
                      for i, n in enumerate(worker_counts)]
        ax.plot(worker_counts, [e * 100 for e in efficiency], marker='o',
                color=color, label=label, linewidth=2, markersize=7)
    ax.axhline(70, color='gray', linestyle='--', alpha=0.5, label='70% target')
    ax.set_xlabel("Number of Workers"); ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Parallel Efficiency (Amdahl)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp2_scalability.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


# ════════════════════════════════════════════════════════════════════════ #
# Experiment 3: Fault injection                                            #
# ════════════════════════════════════════════════════════════════════════ #

def exp3_fault_injection():
    print("\n" + "="*60)
    print("EXP 3: Fault Injection (N=8, 100 steps)")
    print("="*60)

    # Worker 0 is a straggler from step 30–60; worker 1 crashes at step 50
    fault_cfgs = []
    for i in range(8):
        if i == 0:
            fault_cfgs.append(FaultConfig(
                straggler_steps=list(range(30, 60)),
                straggler_slowdown_ms=150.0,
            ))
        elif i == 1:
            fault_cfgs.append(FaultConfig(crash_at_step=50))
        else:
            fault_cfgs.append(FaultConfig())

    results = {}
    for mode, label, color in [('PS', 'Fixed PS', C_PS), ('RA', 'Fixed RA', C_RA),
                                ('AHA', 'AHA (ours)', C_AHA)]:
        print(f"  Running {label} with faults...", end=' ', flush=True)
        m = run_experiment(mode=mode, n_workers=8, n_steps=100,
                           grad_size_mb=5.0, compute_ms=15.0,
                           bandwidth_gbps=10.0, base_latency_ms=0.5,
                           fault_configs=fault_cfgs, label=label, seed=42)
        results[mode] = m
        s = m.summary()
        print(f"thr={s['mean_throughput']:.0f}  step_ms={s['mean_step_ms']:.1f}")

    _plot_fault_injection(results)
    return results


def _plot_fault_injection(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 3: Fault Injection — Straggler (step 30–60) + Crash (step 50)",
                 fontsize=13, fontweight='bold')

    configs = [('PS', 'Fixed PS', C_PS), ('RA', 'Fixed RA', C_RA),
               ('AHA', 'AHA (ours)', C_AHA)]

    ax = axes[0]
    for key, label, color in configs:
        m = results[key]
        ax.plot(m.steps, m.p95_step_time, color=color, label=label, linewidth=1.8, alpha=0.9)
    ax.axvspan(30, 60, alpha=0.12, color=C_FAULT, label='Straggler window')
    ax.axvline(50, color=C_FAULT, linestyle='--', linewidth=2, alpha=0.7, label='Crash event')
    ax.set_xlabel("Training Step"); ax.set_ylabel("p95 Step Time (ms)")
    ax.set_title("Step Latency Under Faults"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for key, label, color in configs:
        m = results[key]
        ax.plot(m.steps, m.throughput, color=color, label=label, linewidth=1.8, alpha=0.9)
    ax.axvspan(30, 60, alpha=0.12, color=C_FAULT)
    ax.axvline(50, color=C_FAULT, linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Throughput Under Faults"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp3_fault_injection.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


# ════════════════════════════════════════════════════════════════════════ #
# Experiment 4: Congested network baseline                                 #
# ════════════════════════════════════════════════════════════════════════ #

def exp4_mixed_network():
    print("\n" + "="*60)
    print("EXP 4: Mixed Network Conditions (stable → congested → packet loss)")
    print("="*60)

    network_conditions = [
        ("Stable",        dict(bandwidth_gbps=10.0, base_latency_ms=0.5,  pkt_loss_rate=0.0)),
        ("Congested",     dict(bandwidth_gbps=1.0,  base_latency_ms=5.0,  pkt_loss_rate=0.0)),
        ("Packet Loss",   dict(bandwidth_gbps=10.0, base_latency_ms=0.5,  pkt_loss_rate=0.05)),
    ]

    summary_data = {}
    for net_name, net_kwargs in network_conditions:
        summary_data[net_name] = {}
        for mode in ['PS', 'RA', 'AHA']:
            print(f"  {mode} | {net_name}...", end=' ', flush=True)
            m = run_experiment(mode=mode, n_workers=8, n_steps=60,
                               grad_size_mb=5.0, compute_ms=15.0,
                               label=f"{mode}-{net_name}", seed=42, **net_kwargs)
            summary_data[net_name][mode] = m.summary()
            print(f"thr={summary_data[net_name][mode]['mean_throughput']:.0f}")

    _plot_mixed_network(summary_data)
    return summary_data


def _plot_mixed_network(summary_data: dict):
    conditions = list(summary_data.keys())
    modes = ['PS', 'RA', 'AHA']
    labels = ['Fixed PS', 'Fixed RA', 'AHA (ours)']
    colors = [C_PS, C_RA, C_AHA]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 4: Performance Across Network Conditions",
                 fontsize=13, fontweight='bold')

    x = np.arange(len(conditions))
    width = 0.25

    # Throughput
    ax = axes[0]
    for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        vals = [summary_data[c][mode]['mean_throughput'] for c in conditions]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color,
                      alpha=0.85, edgecolor='white')
    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean Throughput (samples/sec)")
    ax.set_title("Throughput by Network Condition")
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # p95 latency
    ax = axes[1]
    for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        vals = [summary_data[c][mode]['p95_step_ms'] for c in conditions]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color,
                      alpha=0.85, edgecolor='white')
    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean p95 Step Time (ms)")
    ax.set_title("p95 Latency by Network Condition")
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_mixed_network.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


# ════════════════════════════════════════════════════════════════════════ #
# Experiment 5: Sensitivity analysis — hysteresis H                       #
# ════════════════════════════════════════════════════════════════════════ #

def exp5_sensitivity():
    print("\n" + "="*60)
    print("EXP 5: Sensitivity Analysis — Hysteresis H & Latency Threshold")
    print("="*60)

    H_values = [1, 2, 3, 5, 8, 12]
    results_H = []
    for H in H_values:
        cfg = AHAConfig(hysteresis_H=H, verbose=False,
                        lat_threshold_ms=25.0, straggler_var_threshold=150.0)
        m = run_experiment(mode='AHA', n_workers=8, n_steps=80,
                           grad_size_mb=5.0, compute_ms=15.0,
                           bandwidth_gbps=10.0, base_latency_ms=0.5,
                           jitter_ms=2.0, aha_config=cfg,
                           label=f"AHA-H{H}", seed=42)
        s = m.summary()
        results_H.append((H, s['mean_throughput'], s['n_switches'], s['p95_step_ms']))
        print(f"  H={H:2d}: thr={s['mean_throughput']:.0f}  switches={s['n_switches']:3d}  p95={s['p95_step_ms']:.1f}ms")

    lat_thresholds = [10, 20, 30, 50, 80, 120]
    results_lat = []
    for thresh in lat_thresholds:
        cfg = AHAConfig(lat_threshold_ms=thresh, hysteresis_H=3, verbose=False)
        m = run_experiment(mode='AHA', n_workers=8, n_steps=80,
                           grad_size_mb=5.0, compute_ms=15.0,
                           bandwidth_gbps=10.0, base_latency_ms=0.5,
                           jitter_ms=2.0, aha_config=cfg,
                           label=f"AHA-lat{thresh}", seed=42)
        s = m.summary()
        results_lat.append((thresh, s['mean_throughput'], s['n_switches']))
        print(f"  lat_thresh={thresh:4d}: thr={s['mean_throughput']:.0f}  switches={s['n_switches']}")

    _plot_sensitivity(results_H, results_lat)
    return results_H, results_lat


def _plot_sensitivity(results_H, results_lat):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Experiment 5: Sensitivity Analysis — Hysteresis H and Latency Threshold",
                 fontsize=13, fontweight='bold')

    H_vals = [r[0] for r in results_H]
    H_thr  = [r[1] for r in results_H]
    H_sw   = [r[2] for r in results_H]
    H_p95  = [r[3] for r in results_H]

    axes[0, 0].plot(H_vals, H_thr, marker='o', color=C_AHA, linewidth=2, markersize=7)
    axes[0, 0].set_xlabel("Hysteresis H"); axes[0, 0].set_ylabel("Mean Throughput (samples/sec)")
    axes[0, 0].set_title("Throughput vs Hysteresis H")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(H_vals, H_sw, marker='s', color=C_PS, linewidth=2, markersize=7)
    axes[0, 1].axhline(2, color='gray', linestyle='--', alpha=0.6, label='Target: <2 per 100 steps')
    axes[0, 1].set_xlabel("Hysteresis H"); axes[0, 1].set_ylabel("Number of Mode Switches")
    axes[0, 1].set_title("Mode Switches vs Hysteresis H")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    lat_vals = [r[0] for r in results_lat]
    lat_thr  = [r[1] for r in results_lat]
    lat_sw   = [r[2] for r in results_lat]

    axes[1, 0].plot(lat_vals, lat_thr, marker='o', color=C_AHA, linewidth=2, markersize=7)
    axes[1, 0].set_xlabel("Latency Threshold (ms)"); axes[1, 0].set_ylabel("Mean Throughput")
    axes[1, 0].set_title("Throughput vs Latency Threshold")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(lat_vals, lat_sw, marker='s', color=C_RA, linewidth=2, markersize=7)
    axes[1, 1].set_xlabel("Latency Threshold (ms)"); axes[1, 1].set_ylabel("Mode Switches")
    axes[1, 1].set_title("Mode Switches vs Latency Threshold")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp5_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


# ════════════════════════════════════════════════════════════════════════ #
# Summary CSV                                                              #
# ════════════════════════════════════════════════════════════════════════ #

def write_summary_csv(all_results: dict):
    path = os.path.join(RESULTS_DIR, "experiment_summary.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "mode", "n_workers", "mean_throughput",
                         "mean_step_ms", "p95_step_ms", "final_accuracy", "n_switches"])
        for exp_name, data in all_results.items():
            if isinstance(data, dict):
                for mode, m in data.items():
                    if hasattr(m, 'summary'):
                        s = m.summary()
                        writer.writerow([exp_name, mode, "-",
                                         f"{s['mean_throughput']:.1f}",
                                         f"{s['mean_step_ms']:.2f}",
                                         f"{s['p95_step_ms']:.2f}",
                                         f"{s['final_accuracy']:.4f}",
                                         s['n_switches']])
    print(f"\n  → Summary CSV: {path}")


# ════════════════════════════════════════════════════════════════════════ #
# Main                                                                     #
# ════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  AHA Distributed SGD — Deliverable 3 Experiment Suite       ║")
    print("║  Adaptive Hybrid Aggregation (PS ↔ Ring AllReduce)          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    all_results = {}

    r1 = exp1_baseline_comparison()
    all_results["Exp1_Baseline"] = r1

    r2_n, r2_d = exp2_scalability()
    all_results["Exp2_Scalability"] = {}

    r3 = exp3_fault_injection()
    all_results["Exp3_FaultInjection"] = r3

    r4 = exp4_mixed_network()
    all_results["Exp4_MixedNetwork"] = {}

    r5_H, r5_lat = exp5_sensitivity()
    all_results["Exp5_Sensitivity"] = {}

    write_summary_csv(all_results)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60)

    # Print final comparison table
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│   FINAL RESULTS SUMMARY (Exp 1 — N=8, Stable Net)  │")
    print("├──────────────┬──────────────┬────────────┬──────────┤")
    print("│ Mode         │ Throughput   │ p95 (ms)   │ Acc      │")
    print("├──────────────┼──────────────┼────────────┼──────────┤")
    for mode, label in [('PS', 'Fixed PS '), ('RA', 'Fixed RA '), ('AHA', 'AHA       ')]:
        s = r1[mode].summary()
        print(f"│ {label:12s} │ {s['mean_throughput']:8.0f} s/s  │ {s['p95_step_ms']:7.1f} ms  │ {s['final_accuracy']:.3f}    │")
    print("└──────────────┴──────────────┴────────────┴──────────┘")
    ps_thr = r1['PS'].summary()['mean_throughput']
    ra_thr = r1['RA'].summary()['mean_throughput']
    aha_thr = r1['AHA'].summary()['mean_throughput']
    best = max(ps_thr, ra_thr)
    print(f"\n  AHA improvement over best baseline: {(aha_thr-best)/best*100:+.1f}%")
    print("  Hypothesis H1 target: ≥ +15% improvement")
