# AHA — Adaptive Hybrid Aggregation

> **Parallel and Distributed Computing · BSCS-13D**
> Abdul Moiz (464265) · Aytesam Suhail (454031) · Hassan Aamir (453976)

AHA is a runtime control layer for distributed SGD training that dynamically switches between two communication backends — **Parameter Server (PS)** and **Ring AllReduce (RA)** — based on live telemetry. Instead of committing to one architecture at deployment time, AHA monitors network conditions, straggler behaviour, and gradient traffic every 10 steps and picks whichever backend will be faster right now.

---

## Results at a Glance

| Workers | Fixed PS | Fixed RA |   **AHA** |  AHA vs PS |
| ------: | -------: | -------: | --------: | ---------: |
|       4 |    2,373 |    1,377 |     2,330 |      −1.8% |
|       8 |    4,553 |    1,088 |     3,542 |     −22.2% |
|      16 |    4,502 |      849 |     4,500 |      ≈0.0% |
|  **32** |    4,575 |      496 | **8,319** | **+81.8%** |

_Throughput in samples/sec · Stable 10 GbE network · 5 MB fp32 gradients_

Under fault injection (straggler + crash, N=8), AHA retains **92.5%** of fault-free throughput. Fixed RA retains only **27.1%**.

---

## Project Structure

```
pda_project/
├── core/
│   ├── __init__.py
│   ├── telemetry.py        # TelemetryBus, MetricsLogger, StepTelemetry
│   ├── backends.py         # ParameterServer, RingAllReduce, NetworkLink
│   ├── aha_controller.py   # AHAController, AHAConfig (the novel contribution)
│   └── worker.py           # Worker, SimulatedModel, FaultConfig
├── results/                # Generated plots and CSV (after running experiments)
├── run_experiments.py      # Main experiment runner — all 5 experiments
├── make_presentation.js    # Generates the final defense PPTX
├── make_qa_doc.js          # Generates the Q&A explainer DOCX
└── requirements.txt
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install numpy matplotlib
npm install -g pptxgenjs docx   # only needed for the presentation/doc generators
```

### 2 — Run all experiments

```bash
python run_experiments.py
```

This runs all five experiments and writes plots + a summary CSV to `results/`.  
Expected runtime: **5–15 minutes** depending on machine speed.

### 3 — Generate the presentation

```bash
node make_presentation.js
```

Reads images from `results/` and produces `AHA_Final_Defense_Presentation.pptx` in the project root.

### 4 — Generate the Q&A explainer

```bash
node make_qa_doc.js
```

Produces `AHA_Project_QA_Explainer.docx` in the project root.

---

## Experiments

| #   | Name                 | What it tests                                      | Key output                               |
| --- | -------------------- | -------------------------------------------------- | ---------------------------------------- |
| 1   | Baseline comparison  | PS vs RA vs AHA, N=8, stable net                   | Throughput, p95 latency, loss curves     |
| 2   | Scalability sweep    | N = 4, 8, 16, 32                                   | Speedup, parallel efficiency, Amdahl fit |
| 3   | Fault injection      | Straggler (steps 30–60) + crash (step 50)          | Throughput drop, recovery time           |
| 4   | Mixed network        | Stable / Congested / Packet-loss                   | Per-condition throughput and latency     |
| 5   | Sensitivity analysis | Hysteresis H (1–12), latency threshold (10–120 ms) | Optimal H, switch rate, throughput       |

All experiments share identical model, optimiser, learning-rate schedule, global batch size, and random seeds. The only variable is the communication backend.

---

## How AHA Works

### The decision loop (runs every 10 steps)

```
read telemetry → evaluate triggers → apply hysteresis → (maybe) switch mode
```

**Triggers that push toward PS:**

- p95 step latency > `lat_threshold_ms` (default 50 ms)
- Per-worker step-time variance > `straggler_var_threshold` (straggler detected)
- Packet loss estimate > 1%

**Triggers that push toward RA:**

- PS queue depth > threshold (shard overload)
- All signals stable

**Hysteresis:** the candidate mode must persist for H consecutive decision epochs before the switch is committed. This prevents oscillation on noisy telemetry. Default H = 3.

**Mode switch protocol:** before workers begin the next step in the new mode, the controller enforces a synchronisation barrier and writes a checkpoint. This ensures no partial-update inconsistency crosses a mode boundary.

### Key configuration parameters

| Parameter                 | Default | Effect                                          |
| ------------------------- | ------- | ----------------------------------------------- |
| `hysteresis_H`            | 3       | Epochs a trigger must persist before switching  |
| `decision_epoch_steps`    | 10      | Steps between controller evaluations            |
| `lat_threshold_ms`        | 50      | p95 latency above which PS is preferred         |
| `straggler_var_threshold` | 200     | Step-time variance indicating a straggler       |
| `tau`                     | 2       | Max staleness iterations accepted by PS (τ-BSP) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  T1  Workers  (W0 … WN)                             │
│   forward → backward → emit telemetry               │
└──────────────┬──────────────────────────────────────┘
               │ StepTelemetry (per step)
               ▼
┌─────────────────────────────────────────────────────┐
│  T2  AHA Controller                                 │
│   read telemetry → threshold + hysteresis → token   │
└───────┬─────────────────────────┬───────────────────┘
        │ mode = "PS"             │ mode = "RA"
        ▼                         ▼
┌──────────────┐         ┌──────────────────────┐
│  T3a  PS     │         │  T3b  Ring AllReduce  │
│  Sharded     │         │  Scatter-reduce       │
│  gRPC + τ-BSP│         │  + All-gather         │
└──────┬───────┘         └──────────┬────────────┘
       └──────────┬─────────────────┘
                  ▼
┌─────────────────────────────────────────────────────┐
│  T4  Model Store  (checkpoint on every switch)      │
└─────────────────────────────────────────────────────┘
```

---

## Communication Cost Model

| Mode            | Per-worker bytes/step | Critical path                        | Bottleneck        |
| --------------- | --------------------- | ------------------------------------ | ----------------- |
| PS              | 2G (push + pull)      | O(1) shard hops                      | Shard inbound BW  |
| RA              | 2(N−1)/N × G          | O(N) ring hops                       | Slowest worker    |
| Switch overhead | Negligible            | 1 RTT multicast + O(N) ring reassign | Hysteresis window |

At large N, RA's per-worker cost approaches 2G — identical to PS — but with perfectly balanced load. PS shard inbound bandwidth grows as NG/S; AHA switches to RA before this becomes the bottleneck.

---

## Fault Handling

| Fault              | Detection               | AHA Response                                       |
| ------------------ | ----------------------- | -------------------------------------------------- |
| Worker crash       | Heartbeat timeout (2 s) | Remove from ring, rebuild with N−1, fallback to PS |
| Straggler          | step_var > threshold    | Switch to PS; τ-BSP buffers slow worker            |
| Bandwidth throttle | p95 latency spike       | Prefer PS (avoids ring serialisation)              |
| Packet loss        | pkt_loss_est > 1%       | Prefer PS; gradient checksums enabled              |
| PS shard failure   | gRPC health check       | Remap keys to backup; checkpoint reload            |

---

## Reproducing the Results

All experiments use `seed=42` by default. To reproduce exactly:

```bash
python run_experiments.py   # writes results/ with all PNGs and experiment_summary.csv
```

To re-run a single experiment in isolation:

```python
from run_experiments import exp1_baseline_comparison
results = exp1_baseline_comparison()
```

Each experiment function returns the `MetricsLogger` objects so you can inspect raw data programmatically.

---

## Dependencies

```
numpy
matplotlib
```

No PyTorch or GPU required. The simulation models communication cost explicitly via calibrated sleep calls scaled to gradient size and link bandwidth — no real network hardware needed.

---

## Literature

| Paper                     | Contribution                               | Limitation addressed                  |
| ------------------------- | ------------------------------------------ | ------------------------------------- |
| Dean et al., NeurIPS 2012 | DistBelief — first large-scale PS          | High staleness, no adaptation         |
| Li et al., OSDI 2014      | Parameter Server framework                 | Hotspot risk under skewed access      |
| Goyal et al., 2017        | Linear LR scaling rule for large-batch SGD | Assumes stable synchronous training   |
| Sergeev & Del Balso, 2018 | Horovod (Ring AllReduce)                   | Straggler sensitivity                 |
| Lin et al., ICLR 2018     | Deep Gradient Compression                  | Compression may harm stability        |
| Stich, JMLR 2019          | Local SGD                                  | Extra drift, no runtime policy        |
| Jiang et al., OSDI 2020   | BytePS                                     | Still centralised under heavy traffic |
| Zhong et al., MLSys 2021  | Bagua (multi-mode)                         | Policy selection not automated        |

AHA's contribution: the first reproducible, telemetry-driven runtime controller with bounded-staleness consistency guards benchmarked across five network/fault regimes.
