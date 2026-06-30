import os
import sys
import json
import time
import datetime
import subprocess
import threading
import queue
import numpy as np
import onnxruntime as ort

from .features import FeaturePreprocessor
from .utils import (
    get_container_cgroup_info,
    get_container_metadata,
    get_container_memory_usage,
    get_container_memory_limit,
    calculate_regret,
    calculate_trigger_threshold,
    pause_container,
    get_container_memory_stats
)

def load_config(config_path=None):
    """
    Search paths order for the configuration:
    1. config_path parameter if specified
    2. current directory ./doom_killer_config.json (or ./psi_guard_config.json)
    3. config/doom_killer_config.json relative to project root
    """
    search_paths = []
    if config_path:
        search_paths.append(config_path)
    
    search_paths.append("doom_killer_config.json")
    search_paths.append("psi_guard_config.json")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    search_paths.append(os.path.join(project_root, "config", "doom_killer_config.json"))
    search_paths.append(os.path.join(project_root, "config", "psi_guard_config.json"))
    search_paths.append(os.path.join(project_root, "doom_killer_config.json"))
    search_paths.append(os.path.join(project_root, "psi_guard_config.json"))
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    config = json.load(f)
                    print(f"Loaded configuration from: {os.path.abspath(path)}")
                    return config
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
                
    print("Config file not found. Using default values.")
    return {
        "target_container": "doom-target",
        "model_path": os.path.join(project_root, "models", "doom_model.onnx"),
        "action": "pause",
        "poll_interval": 1.0,
        "anomaly_threshold_base": 0.05,
        "anomaly_threshold_max": 0.12,
        "scaling_factor_k": 0.001,
        "priority_default": 50,
        "weights": {
            "uptime": 0.001,
            "priority": 0.4,
            "memory": 0.01
        }
    }

def enqueue_output(out, q):
    for line in iter(out.readline, ''):
        q.put(line)
    out.close()

def run_daemon(config_path=None, model_path_override=None, target_override=None):
    """
    Monitors a running Docker container using eBPF and predicts OOM using ONNX runtime.
    Freezes the container if OOM is predicted soon.
    """
    if os.geteuid() != 0:
        print("Error: must be run as root (sudo)")
        sys.exit(1)

    print("Starting DOOM-Killer...")

    config = load_config(config_path)
    target = target_override if target_override else config.get("target_container", "doom-target")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path = model_path_override if model_path_override else config.get("model_path", "doom_model.onnx")
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        possible_paths = [
            os.path.join(project_root, "models", model_path),
            os.path.join(project_root, model_path)
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: model file {model_path} not found")
        sys.exit(1)
        
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # IsolationForest outputs ['label', 'scores']. We want 'scores' which corresponds to decision_function.
    output_names = [o.name for o in session.get_outputs()]
    output_name = 'scores' if 'scores' in output_names else output_names[0]
    
    print(f"Waiting for target container: {target}")
    pid, cgroup_path, full_cgroup_path = None, None, None
    while True:
        pid, cgroup_path, full_cgroup_path = get_container_cgroup_info(target)
        if pid:
            break
        time.sleep(2)
        
    print(f"Target container detected (PID: {pid})")
    print(f"Cgroup: {full_cgroup_path}")

    # Build and run the bpftrace subprocess
    ebpf_script = f'tracepoint:kmem:mm_page_alloc /cgroup == cgroupid("{full_cgroup_path}")/ {{ @allocations = count(); }} interval:s:1 {{ print(@allocations); clear(@allocations); }}'
    print("Attaching eBPF probe...")
    
    sensor = subprocess.Popen(
        ["stdbuf", "-oL", "bpftrace", "-e", ebpf_script],
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1
    )
    
    # Read bpftrace stdout asynchronously in a separate thread to avoid blocking issues
    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(sensor.stdout, q))
    t.daemon = True
    t.start()
    
    preprocessor = FeaturePreprocessor()
    
    print("Monitoring active")
    
    mem_limit_bytes = get_container_memory_limit(full_cgroup_path)
    if not mem_limit_bytes or mem_limit_bytes == 0:
        mem_limit_bytes = 256 * 1024 * 1024
        print("Warning: no memory limit detected, using default 256MB scale")
    else:
        print(f"Memory limit: {mem_limit_bytes / (1024 * 1024):.1f}MB")
        
    paused = False
    
    try:
        while sensor.poll() is None:
            try:
                line = q.get(timeout=2.0)
            except queue.Empty:
                continue
                
            line = line.strip()
            if "@allocations:" in line:
                try:
                    velocity = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    continue
                
                # Check container metadata and state
                uptime_sec, priority, already_paused = get_container_metadata(target, config.get("priority_default", 50))
                
                if already_paused:
                    if not paused:
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Container '{target}' is already paused.")
                        paused = True
                    continue
                
                # Query current memory usage from cgroup
                mem_bytes = get_container_memory_usage(full_cgroup_path)
                mem_mb = mem_bytes / (1024 * 1024)
                
                # Query memory.stat fields
                pgmajfault, anon, file_mem = get_container_memory_stats(full_cgroup_path)

                # Compute scale-invariant features
                features = preprocessor.get_features(velocity, mem_bytes, mem_limit_bytes, pgmajfault, anon, file_mem)
                
                # Predict anomaly score using ONNX (sign convention: anomaly_score = -decision_score)
                input_data = np.array([features], dtype=np.float32)
                out = session.run([output_name], {input_name: input_data})
                decision_score = float(out[0][0][0])
                anomaly_score = -decision_score
                
                # Regret calculations
                regret = calculate_regret(uptime_sec, priority, mem_mb, config.get("weights", {}))
                trigger_threshold = calculate_trigger_threshold(regret, config)
                
                # Log metrics cleanly
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                      f"allocs/s: {velocity:6d} | "
                      f"mem: {mem_mb:6.1f}MB | "
                      f"regret: {regret:5.1f} | "
                      f"thresh: {trigger_threshold:6.4f} | "
                      f"anomaly: {anomaly_score:6.4f}")
                      
                # Evaluate Intervention
                failsafe_trigger = False
                if mem_limit_bytes and mem_bytes >= 0.97 * mem_limit_bytes and velocity > 5000:
                    print(f"\n[FAIL-SAFE] Memory critical ({mem_mb:.1f}MB / {mem_limit_bytes / (1024 * 1024):.1f}MB) | allocs/s: {velocity} (overriding prediction)")
                    failsafe_trigger = True
                    
                if failsafe_trigger or (anomaly_score > trigger_threshold):
                    success = pause_container(target)
                    if success:
                        paused = True
                        break
                        
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("Cleaning up sensor...")
        sensor.terminate()
        sensor.wait()
        print("Done")
