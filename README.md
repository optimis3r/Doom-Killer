# DOOM-Killer: Unsupervised OOM Anomaly Mitigation Engine

DOOM-Killer is an unsupervised, scale-invariant Out-Of-Memory (OOM) mitigation framework designed for containerized Linux workloads. By hooking physical memory allocation rates at the kernel level using eBPF probes and reading memory statistics from cgroup v2, the engine tracks container footprints in real-time, detects anomalous behavior using a trained Isolation Forest model (deployed via ONNX Runtime), and performs proactive container mitigation (Cgroup Freezer freezing) to prevent chaotic OOM-kills.

---

## Project Structure

```
proactiveOOM/
├── config/
│   └── doom_killer_config.json        # Core daemon config (limits, weights, triggers)
├── data/
│   ├── healthyTelemetry.csv           # Telemetry training dataset from healthy workloads
│   └── crashValidationData.csv        # Labeled crash episodes for model validation
├── docs/
│   ├── PSI-Gaurd-roadmap.pdf          # Technical roadmap
│   └── Proactive_OOM_Killer_Study_Guide.pdf # Reference documentation
├── models/
│   └── doom_model.onnx                # Compiled Isolation Forest anomaly detection model
├── src/
│   ├── __init__.py
│   ├── daemon.py                      # eBPF monitoring & ONNX inference daemon
│   ├── data_factory.py                # Telemetry capture & workload simulation factory
│   ├── features.py                    # Scale-invariant feature engineering preprocessor
│   ├── train.py                       # Isolation Forest training, LOWO validation, & ONNX export logic
│   └── utils.py                       # Docker metadata, cgroup parser, and actuator functions
├── target_app/
│   ├── flask_app/
│   │   ├── app.py                     # Flask server with healthy and leaking endpoints
│   │   └── Dockerfile                 # Docker config for the Flask container
│   └── compute_job/
│       ├── job.py                     # Non-leaking matrix multiplication batch job script
│       └── Dockerfile                 # Docker config for the compute-bound container
├── .gitignore                         # Project VCS exclusions
├── doom_killer.py                     # Unified Command Line Interface (CLI)
└── requirements.txt                   # Python package dependencies
```

---

## Prerequisites and Setup

### System Requirements
1. **Linux OS**: eBPF probes require a Linux kernel with `CONFIG_BPF`, `CONFIG_BPF_SYSCALL`, and `CONFIG_TRACING` enabled.
2. **Docker**: Used to run target workloads.
3. **bpftrace**: The command line utility to execute the eBPF tracepoint scripts.
   ```bash
   sudo apt-get install -y bpftrace stdbuf
   ```
4. **Apache Bench (ab)**: Used by the data factory for load generation.
   ```bash
   sudo apt-get install -y apache2-utils
   ```

### Python Virtual Environment Setup
1. Activate the workspace virtual environment:
   ```bash
   source .venv/bin/python
   ```
2. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start (Running the Daemon)

To start protecting a container using the pre-trained Isolation Forest model, run the DOOM-Killer mitigation daemon. The daemon polls container memory metrics, runs real-time inferences against the ONNX model, and pauses the target container if the anomaly score exceeds the dynamic regret-based threshold.

Note: The daemon must be run as root (`sudo`) to attach eBPF probes.

```bash
sudo .venv/bin/python doom_killer.py run --config config/doom_killer_config.json
```

### Arguments
*   `--config` (default: searches CWD and project root fallbacks): Configuration JSON file path.
*   `--model` (optional override): Overrides the path to the ONNX model file.
*   `--target` (optional override): Overrides the target container name to monitor (default: `doom-target`).

---

## Developer Guide (Data Harvesting and Model Retraining)

Follow these steps if you want to collect new telemetry data and retrain the Isolation Forest model on your custom workloads.

### 1. Healthy Telemetry Harvesting (Training Data)
Runs a series of simulation runs across different memory limit tiers for real workloads (Postgres, Redis, numpy compute, Flask) under realistic load. The collected telemetry is clean and OOMs are avoided.
Note: Must be run as root (`sudo`).

```bash
sudo .venv/bin/python doom_killer.py harvest-healthy --runs 3 --duration 300 --output data/healthyTelemetry.csv
```
*   `--runs` (default: `3`): The number of runs per workload type.
*   `--duration` (default: `300`): The length of each run in seconds.
*   `--workload` (default: `all`): Workload to run (`postgres`, `redis`, `compute`, `flask`, or `all`).
*   `--output` (default: `data/healthyTelemetry.csv`): Path to output CSV file.

### 2. Workload Telemetry Crash Simulation (Validation Data)
Boots the leaking Flask server container with a random memory limit, triggers traffic to induce OOM, and records telemetry backward-labeled from the OOM crash moment (crash is at TTO = 0 seconds). Used only for model threshold calibration and validation.
Note: Must be run as root (`sudo`).

```bash
sudo .venv/bin/python doom_killer.py generate-data --runs 5 --output data/crashValidationData.csv
```
*   `--runs` (default: `100`): The number of OOM runs.
*   `--output` (default: `data/crashValidationData.csv`): Path to output validation CSV file.

### 3. Anomaly Detection Model Training
Reads the healthy telemetry, runs scale-invariant feature extraction (remaining headroom, relative velocity/acceleration, major page faults, cache-to-RSS, and rolling windows), performs leave-one-workload-out validation, and exports the final model to ONNX.

```bash
.venv/bin/python doom_killer.py train --data data/healthyTelemetry.csv --output models/doom_model.onnx
```
*   `--data` (default: `data/healthyTelemetry.csv`): Input CSV file path.
*   `--output` (default: `models/doom_model.onnx`): Output ONNX model file path.

---

## Architectural Mechanics

### 1. Scale-Invariant Features
Raw byte values are bad features for machine learning as containers have widely different memory limits. Features are scaled relative to container limits:
- **Remaining Memory Ratio**: (LimitBytes - MemoryUsage) / LimitBytes
- **Relative Velocity**: Allocation speed per second scaled to limit size.
- **Relative Acceleration**: Rate of change of relative velocity.
- **Major Page Fault Rate**: Diff in pgmajfault per second.
- **Cache-to-RSS Ratio**: Ratio of file-backed cache to anonymous resident memory (file / anon).

### 2. Regret-Cost Threshold Scaling
Instead of a fixed threshold, DOOM-Killer computes a dynamic threshold using "regret":
$$Regret = w_1 \cdot Uptime + w_2 \cdot Priority - w_3 \cdot Memory$$
$$Threshold = Base + k \cdot Regret$$
This ensures:
- Containers running for a long time or labeled with high priority have higher trigger thresholds (we delay mitigation as much as possible because interrupting them has high cost).
- Low priority or heavy memory-intensive containers trigger early mitigation.

