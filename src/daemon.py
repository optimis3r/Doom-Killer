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

"""
    Search paths order for the configuration:
    1. config_path parameter (need to specify)
    2. current directory ./doom_killer_config.json
    3. config/doom_killer_config.json relative to project root
"""

def load_config(config_path=None):
    search_paths = []
    if config_path:
        search_paths.append(config_path)
    
    search_paths.append("doom_killer_config.json")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    search_paths.append(os.path.join(project_root, "config", "doom_killer_config.json"))
    search_paths.append(os.path.join(project_root, "doom_killer_config.json"))
    
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

"""
    Monitoring running Docker container using eBPF and predicting OOM using ONNX runtime.
    Freezes container if OOM is predicted soon.
"""

def get_running_containers():
    """
    Returns a list of running Docker container names.
    """
    try:
        res = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
        names = [n.strip() for n in res.stdout.splitlines() if n.strip()]
        return names
    except Exception as e:
        print(f"Warning: Failed to get running Docker containers: {e}")
        return []

def run_daemon(config_path=None, model_path_override=None, target_override=None):
    if os.geteuid() != 0:
        print("Error: must run as root (sudo)")
        sys.exit(1)

    print("Starting DOOM-Killer...")

    config = load_config(config_path)
    target = target_override if target_override else config.get("target_container", "doom-target")
    auto_discovery = (target.lower() in ["all", "auto", "any"])
    
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
    
    # IsolationForest will output ['label', 'scores']. We want 'scores' which corresponds to decision_function.
    output_names = [o.name for o in session.get_outputs()]
    output_name = 'scores' if 'scores' in output_names else output_names[0]
    
    active_trackers = {}
    poll_interval = float(config.get("poll_interval", 1.0))
    
    if auto_discovery:
        print("Mode: Host-wide Auto-discovery (monitoring all running containers)")
    else:
        print(f"Mode: Single Target (monitoring container: {target})")
        
    try:
        while True:
            # 1. Resolve current active container targets
            if auto_discovery:
                current_targets = get_running_containers()
            else:
                # Check if target is running
                pid, cgroup_path, full_cgroup_path = get_container_cgroup_info(target)
                current_targets = [target] if pid else []
                
            # Stop trackers for containers that are no longer running
            for name in list(active_trackers.keys()):
                if name not in current_targets:
                    print(f"Stopped monitoring: {name}")
                    tracker = active_trackers[name]
                    tracker["sensor"].terminate()
                    tracker["sensor"].wait()
                    del active_trackers[name]
                    
            # Initialize trackers for newly detected containers
            for name in current_targets:
                if name not in active_trackers:
                    uptime_sec, priority, already_paused = get_container_metadata(name, config.get("priority_default", 50))
                    if already_paused:
                        continue
                        
                    pid, cgroup_path, full_cgroup_path = get_container_cgroup_info(name)
                    if not pid:
                        continue
                        
                    ebpf_script = f'tracepoint:kmem:mm_page_alloc /cgroup == cgroupid("{full_cgroup_path}")/ {{ @allocations = count(); }} interval:s:1 {{ print(@allocations); clear(@allocations); }}'
                    try:
                        sensor = subprocess.Popen(
                            ["stdbuf", "-oL", "bpftrace", "-e", ebpf_script],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True,
                            bufsize=1
                        )
                    except Exception as e:
                        print(f"Warning: failed to start eBPF sensor for {name}: {e}")
                        continue
                        
                    q = queue.Queue()
                    t = threading.Thread(target=enqueue_output, args=(sensor.stdout, q))
                    t.daemon = True
                    t.start()
                    
                    mem_limit_bytes = get_container_memory_limit(full_cgroup_path)
                    if not mem_limit_bytes or mem_limit_bytes == 0:
                        mem_limit_bytes = 256 * 1024 * 1024
                        
                    active_trackers[name] = {
                        "pid": pid,
                        "full_cgroup_path": full_cgroup_path,
                        "sensor": sensor,
                        "queue": q,
                        "thread": t,
                        "preprocessor": FeaturePreprocessor(),
                        "mem_limit_bytes": mem_limit_bytes,
                        "paused": False
                    }
                    print(f"Monitoring container: {name} (limit={mem_limit_bytes / (1024*1024):.1f}MB)")
                    
            # 2. Process metrics and inference for all running trackers
            for name, tracker in list(active_trackers.items()):
                if tracker["sensor"].poll() is not None:
                    print(f"Warning: sensor for {name} died. Removing.")
                    del active_trackers[name]
                    continue
                    
                # Read all lines from queue to get the latest allocation velocity
                velocity = 0
                while True:
                    try:
                        line = tracker["queue"].get_nowait()
                        line = line.strip()
                        if "@allocations:" in line:
                            try:
                                velocity = int(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                pass
                    except queue.Empty:
                        break
                        
                uptime_sec, priority, already_paused = get_container_metadata(name, config.get("priority_default", 50))
                if already_paused:
                    if not tracker["paused"]:
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{name}] Container is paused.")
                        tracker["paused"] = True
                    continue
                    
                full_path = tracker["full_cgroup_path"]
                mem_bytes = get_container_memory_usage(full_path)
                mem_mb = mem_bytes / (1024 * 1024)
                mem_limit_bytes = tracker["mem_limit_bytes"]
                
                pgmajfault, anon, file_mem = get_container_memory_stats(full_path)
                
                features = tracker["preprocessor"].get_features(
                    velocity, mem_bytes, mem_limit_bytes, pgmajfault, anon, file_mem
                )
                
                input_data = np.array([features], dtype=np.float32)
                out = session.run([output_name], {input_name: input_data})
                decision_score = float(out[0][0][0])
                anomaly_score = -decision_score
                
                regret = calculate_regret(uptime_sec, priority, mem_mb, config.get("weights", {}))
                trigger_threshold = calculate_trigger_threshold(regret, config)
                
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{name}] "
                      f"allocs/s: {velocity:6d} | "
                      f"mem: {mem_mb:6.1f}MB | "
                      f"regret: {regret:5.1f} | "
                      f"thresh: {trigger_threshold:6.4f} | "
                      f"anomaly: {anomaly_score:6.4f}")
                      
                failsafe_trigger = False
                if mem_limit_bytes and mem_bytes >= 0.97 * mem_limit_bytes and velocity > 5000:
                    print(f"\n[FAIL-SAFE] [{name}] Memory critical ({mem_mb:.1f}MB / {mem_limit_bytes / (1024 * 1024):.1f}MB) | allocs/s: {velocity} (overriding prediction)")
                    failsafe_trigger = True
                    
                if failsafe_trigger or (anomaly_score > trigger_threshold):
                    success = pause_container(name)
                    if success:
                        tracker["paused"] = True
                        tracker["sensor"].terminate()
                        tracker["sensor"].wait()
                        del active_trackers[name]
                        
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("Cleaning up sensors...")
        for name, tracker in list(active_trackers.items()):
            try:
                tracker["sensor"].terminate()
                tracker["sensor"].wait()
            except Exception:
                pass
        print("Done")
