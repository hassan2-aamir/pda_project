# PDA Project — Experiment Runner

Quick setup to run the experiments locally.

Prerequisites
- Python 3.8+ (3.10 recommended)
- Git (optional)

Setup
1. Create a virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Run experiments (single script):

```powershell
python run_experiments.py
```

Notes
- Results are written to the `results/` directory as CSV and PNG files.
- I fixed a module-name mismatch: `core/telemetry.py` is the correct module (previously misspelled).
- If you want a quick import check without running experiments:

```powershell
python -c "import core.telemetry, core.backends, core.worker, core.aha_controller; print('imports OK')"
```
