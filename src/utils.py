import os
import subprocess
import json
import time
import datetime

"""
    Retrieves the PID, relative cgroup path, and absolute cgroup v2 path
    for a given running Docker container.
"""

def get_container_cgroup_info(container_name):
    try:
        pid_result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
            capture_output=True, text=True, check=True
        )
        container_pid = pid_result.stdout.strip()
        if container_pid == "0":
            return None, None, None

        # Read /proc/<PID>/cgroup to extract the cgroup v2 path
        cgroup_file = f"/proc/{container_pid}/cgroup"
        if not os.path.exists(cgroup_file):
            return None, None, None
            
        with open(cgroup_file, "r") as f:
            cgroup_content = f.read()

        cgroup_path = ""
        for line in cgroup_content.splitlines():
            if line.startswith("0::"):
                cgroup_path = line.split("::")[1].strip()
                break

        if not cgroup_path:
            return None, None, None

        full_cgroup_path = f"/sys/fs/cgroup{cgroup_path}" if cgroup_path.startswith("/") else f"/sys/fs/cgroup/{cgroup_path}"
        return container_pid, cgroup_path, full_cgroup_path
    except Exception:
        return None, None, None

"""
    Fetches running metadata of the container: Uptime (seconds), priority, and paused state.
"""

def get_container_metadata(container_name, default_priority):
    try:
        res = subprocess.run(
            ["docker", "inspect", "-f", '{"StartedAt": "{{.State.StartedAt}}", "Labels": {{json .Config.Labels}}, "Paused": {{.State.Paused}}}', container_name],
            capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout.strip())
        
        started_at = data.get("StartedAt", "")
        uptime_seconds = 0.0
        if started_at and started_at != "0001-01-01T00:00:00Z":
            t_str = started_at.strip().rstrip('Z')
            if '.' in t_str:
                base, frac = t_str.split('.')
                frac = frac[:6]
                t_str = f"{base}.{frac}"
            dt_start = datetime.datetime.fromisoformat(t_str)
            dt_now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            uptime_seconds = max(0.0, (dt_now - dt_start).total_seconds())
            
        labels = data.get("Labels") or {}
        priority = default_priority
        priority_key = None
        if "doom-killer.priority" in labels:
            priority_key = "doom-killer.priority"
            
        if priority_key:
            try:
                priority = float(labels[priority_key])
            except ValueError:
                pass
                
        paused = data.get("Paused", False)
        return uptime_seconds, priority, paused
    except Exception:
        return 0.0, default_priority, False

def get_container_memory_usage(full_cgroup_path):
    try:
        mem_path = os.path.join(full_cgroup_path, "memory.current")
        if os.path.exists(mem_path):
            with open(mem_path, "r") as f:
                return int(f.read().strip())
        return 0
    except Exception:
        return 0

def get_container_memory_limit(full_cgroup_path):
    try:
        max_path = os.path.join(full_cgroup_path, "memory.max")
        if os.path.exists(max_path):
            with open(max_path, "r") as f:
                content = f.read().strip()
                if content == "max":
                    return None
                return int(content)
        return None
    except Exception:
        return None

def calculate_regret(uptime_seconds, priority, mem_mb, weights):
    w_1 = weights.get("uptime", 0.001)
    w_2 = weights.get("priority", 0.4)
    w_3 = weights.get("memory", 0.01)
    
    regret = (w_1 * uptime_seconds) + (w_2 * priority) - (w_3 * mem_mb)
    return regret

def calculate_trigger_threshold(regret, config):
    base = config.get("anomaly_threshold_base", config.get("tto_threshold_base", 0.05))
    max_safety = config.get("anomaly_threshold_max", config.get("tto_threshold_min", 0.12))
    k = config.get("scaling_factor_k", 0.001)
    
    threshold = base + (k * regret)
    return min(max_safety, threshold)

# acutator will pause and freeze
def pause_container(container_name):
    print(f"\n[ACTUATOR] Freezing {container_name}")
    res = subprocess.run(["docker", "pause", container_name], capture_output=True, text=True)
    if res.returncode == 0:
        print("[ACTUATOR] Frozen")
        return True
    else:
        print(f"[ACTUATOR] Error: failed to pause container: {res.stderr.strip()}")
        return False


def get_container_memory_stats(full_cgroup_path):
    pgmajfault = 0
    anon = 0
    file = 0
    try:
        stat_path = os.path.join(full_cgroup_path, "memory.stat")
        if os.path.exists(stat_path):
            with open(stat_path, "r") as f:
                content = f.read()
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    key, val = parts[0], int(parts[1])
                    if key == "pgmajfault":
                        pgmajfault = val
                    elif key == "anon":
                        anon = val
                    elif key == "file":
                        file = val
    except Exception:
        pass
    return pgmajfault, anon, file



