import subprocess
import time
import csv
import os
import random
import sys
import queue
import threading
from .utils import get_container_memory_stats

def enqueue_output(out, q):
    for line in iter(out.readline, ''):
        q.put(line)
    out.close()

def generate_data(output_path, num_runs=100):
    """
    Orchestrates Docker containers, drives simulated memory-leak traffic using Apache Bench,
    gathers raw allocation velocity using eBPF probes, labels telemetry relative to the OOM event,
    and writes results to a CSV file.
    """
    if os.geteuid() != 0:
        print("Error: must be run as root (sudo)")
        sys.exit(1)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # 0. Check and build Docker image
    print("Checking image...")
    img_check = subprocess.run(["docker", "image", "inspect", "real-web-server"], capture_output=True)
    if img_check.returncode != 0:
        target_app_dir = os.path.join(PROJECT_ROOT, "target_app", "flask_app")
        print("Building real-web-server image...")
        subprocess.run(["docker", "build", "-t", "real-web-server", target_app_dir], check=True)

    labeledDataset = []

    mem_limits = {
        "128M": 128 * 1024 * 1024,
        "192M": 192 * 1024 * 1024,
        "256M": 256 * 1024 * 1024,
        "384M": 384 * 1024 * 1024,
        "512M": 512 * 1024 * 1024
    }

    for run_idx in range(1, num_runs + 1):
        print(f"Run {run_idx}/{num_runs} (Flask leak)...")

        # 1. CLEAR OLD CONTAINERS
        subprocess.run(["docker", "rm", "-f", "doom-target"], capture_output=True)

        # 2. START THE FLASK SERVER WITH RANDOM MEMORY LIMIT
        limit_str = random.choice(list(mem_limits.keys()))
        limit_bytes = mem_limits[limit_str]
        subprocess.run([
            "docker", "run", "-d", "-m", limit_str, "--name", "doom-target", "real-web-server"
        ], check=True)

        # Retrieve container IP address to bypass host iptables/DNAT port-forwarding issues
        ip_result = subprocess.run(
            ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", "doom-target"],
            capture_output=True, text=True, check=True
        )
        container_ip = ip_result.stdout.strip()

        # Give Flask 2 seconds to boot up and bind to the port
        time.sleep(2)

        # 3. GET DOCKER CONTAINER PID AND CGROUP PATH
        pid_result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", "doom-target"],
            capture_output=True, text=True, check=True
        )
        container_pid = pid_result.stdout.strip()

        # Read /proc/<PID>/cgroup to extract the cgroup v2 path
        with open(f"/proc/{container_pid}/cgroup", "r") as f:
            cgroup_content = f.read()

        cgroup_path = ""
        for line in cgroup_content.splitlines():
            if line.startswith("0::"):
                cgroup_path = line.split("::")[1].strip()
                break

        if not cgroup_path:
            print("Error: failed to resolve cgroup")
            continue

        # Build the absolute path to the cgroup directory
        full_cgroup_path = f"/sys/fs/cgroup{cgroup_path}" if cgroup_path.startswith("/") else f"/sys/fs/cgroup/{cgroup_path}"

        # 4. START EBPF SENSOR FILTERED BY CGROUP (Line-Buffered)
        ebpfScript = f'tracepoint:kmem:mm_page_alloc /cgroup == cgroupid("{full_cgroup_path}")/ {{ @allocations = count(); }} interval:s:1 {{ print(@allocations); clear(@allocations); }}'

        print("Starting probe...")
        sensor = subprocess.Popen(
            ["stdbuf", "-oL", "bpftrace", "-e", ebpfScript],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        # Give bpftrace a moment to attach
        time.sleep(2)

        # Introduce some randomness to make the dataset noisier
        # Adjust requests so they scale with memory size to make sure we hit OOM!
        concurrency = random.randint(5, 15)
        # A 512M container needs more requests to OOM than a 128M container
        requests = concurrency * int(2000 * (limit_bytes / (256 * 1024 * 1024)))
        # Ensure a minimum number of requests
        requests = max(requests, concurrency * 1000)

        # 5. FIRE THE HTTP LOAD GENERATOR (Apache Bench)
        print(f"Sending load (concurrency={concurrency}, requests={requests})...")
        load_tester = subprocess.Popen(
            ["ab", "-n", str(requests), "-c", str(concurrency), f"http://{container_ip}:5000/matchmake"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # 6. THE DEATH WATCHER LOOP (Real-time telemetry logging)
        print("Monitoring active")
        oom_killed = False
        runData = []
        
        # Read bpftrace stdout asynchronously in a separate thread to avoid blocking issues
        q = queue.Queue()
        t = threading.Thread(target=enqueue_output, args=(sensor.stdout, q))
        t.daemon = True
        t.start()

        try:
            while True:
                try:
                    line = q.get(timeout=1.5)
                except queue.Empty:
                    line = "@allocations: 0"
                if not line:
                    break
                    
                if "@allocations" in line:
                    try:
                        velocity = int(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        continue
                    
                    # Check if OOM killed
                    oomCheck = subprocess.run(
                        ["docker", "inspect", "-f", "{{.State.OOMKilled}}", "doom-target"],
                        capture_output=True, text=True
                    )
                    if "true" in oomCheck.stdout:
                        print("OOM occurred")
                        oom_killed = True
                        break
                        
                    # Check if running
                    statusCheck = subprocess.run(
                        ["docker", "inspect", "-f", "{{.State.Running}}", "doom-target"],
                        capture_output=True, text=True
                    )
                    if "false" in statusCheck.stdout:
                        print("Error: container stopped without OOM")
                        break
                        
                    # Query current memory usage from cgroup
                    mem_usage = 0
                    try:
                        mem_file = os.path.join(full_cgroup_path, "memory.current")
                        if os.path.exists(mem_file):
                            with open(mem_file, "r") as f:
                                mem_usage = int(f.read().strip())
                    except Exception:
                        pass
                    
                    # Query memory.stat fields
                    pgmajfault, anon, file_mem = get_container_memory_stats(full_cgroup_path)
                    print(f"  mem={mem_usage / (1024*1024):.1f}MB | allocs/s={velocity} | faults={pgmajfault}")
                    runData.append({
                        "Velocity": velocity,
                        "Mem_Usage_Bytes": mem_usage,
                        "Pgmajfault": pgmajfault,
                        "Anon": anon,
                        "File": file_mem
                    })
        except KeyboardInterrupt:
            print("Interrupted")
            break
        finally:
            # 7. HARVEST THE DATA AND CLEAN UP
            print("Cleaning up...")
            sensor.terminate()
            load_tester.terminate()
            sensor.wait()
            load_tester.wait()
            subprocess.run(["docker", "rm", "-f", "doom-target"], capture_output=True)

        if not oom_killed:
            print("Skipped (no OOM)")
            continue

        # Reverse labeling: TTO_Seconds = 0 is the moment of OOM crash
        runData.reverse()

        for i, data in enumerate(runData):
            labeledDataset.append({
                "Run_ID": run_idx,
                "TTO_Seconds": i,
                "Limit_Bytes": limit_bytes,
                "Mem_Usage_Bytes": data["Mem_Usage_Bytes"],
                "Velocity": data["Velocity"],
                "Pgmajfault": data["Pgmajfault"],
                "Anon": data["Anon"],
                "File": data["File"]
            })

    # Write the accumulated data to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Run_ID", "TTO_Seconds", "Limit_Bytes", "Mem_Usage_Bytes", "Velocity", "Pgmajfault", "Anon", "File"])
        writer.writeheader()
        writer.writerows(labeledDataset)

    print(f"Exported: {len(labeledDataset)} rows to {output_path}")


def harvest_healthy(output_path, workload_type="all", duration=300, num_runs=3):
    """
    Runs various healthy (non-leaking) containerized workloads and collects telemetry.
    No OOM is induced.
    """
    if os.geteuid() != 0:
        print("Error: must be run as root (sudo)")
        sys.exit(1)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    mem_limits = {
        "128M": 128 * 1024 * 1024,
        "192M": 192 * 1024 * 1024,
        "256M": 256 * 1024 * 1024,
        "384M": 384 * 1024 * 1024,
        "512M": 512 * 1024 * 1024
    }

    if workload_type == "all":
        workloads = ["postgres", "redis", "compute", "flask"]
    else:
        workloads = [workload_type]

    harvested_dataset = []
    run_id_counter = 1

    for workload in workloads:
        for run_idx in range(1, num_runs + 1):
            # Clear older container if any exists
            subprocess.run(["docker", "rm", "-f", "doom-target"], capture_output=True)

            limit_str = random.choice(list(mem_limits.keys()))
            limit_bytes = mem_limits[limit_str]

            print(f"Run {run_id_counter} ({workload}, limit={limit_str})...")

            if workload == "postgres":
                # Determine shared_buffers dynamically to be safe
                if limit_str == "128M":
                    shared_buffers = "16MB"
                elif limit_str == "192M":
                    shared_buffers = "24MB"
                elif limit_str == "256M":
                    shared_buffers = "32MB"
                else:
                    shared_buffers = "64MB"

                subprocess.run([
                    "docker", "run", "-d", "--name", "doom-target",
                    "-m", limit_str,
                    "-e", "POSTGRES_PASSWORD=postgres",
                    "postgres:alpine",
                    "-c", f"shared_buffers={shared_buffers}"
                ], check=True)

                print("Waiting for Postgres to initialize...")
                initialized = False
                for _ in range(30):
                    check = subprocess.run(["docker", "exec", "doom-target", "pg_isready", "-U", "postgres"], capture_output=True)
                    if check.returncode == 0:
                        initialized = True
                        break
                    time.sleep(1)

                if not initialized:
                    print("Postgres failed to initialize in time. Skipping run...")
                    continue

                # Initialize pgbench schema
                print("Initializing pgbench schema...")
                subprocess.run([
                    "docker", "exec", "doom-target",
                    "pgbench", "-i", "-s", "1", "-U", "postgres", "postgres"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Start pgbench asynchronously to stress the DB for the run duration
                print("Starting asynchronous pgbench workload...")
                subprocess.Popen([
                    "docker", "exec", "doom-target",
                    "pgbench", "-c", "4", "-T", str(duration + 10), "-U", "postgres", "postgres"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            elif workload == "redis":
                subprocess.run([
                    "docker", "run", "-d", "--name", "doom-target",
                    "-m", limit_str,
                    "redis:alpine"
                ], check=True)

                time.sleep(3) # Let redis start up

                # Run redis-benchmark asynchronously for read/write churn
                print("Starting asynchronous redis-benchmark...")
                subprocess.Popen([
                    "docker", "exec", "doom-target",
                    "redis-benchmark", "-c", "10", "-t", "set,get", "-n", "10000000", "--quiet"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            elif workload == "compute":
                # Ensure local compute image is built
                img_check = subprocess.run(["docker", "image", "inspect", "doom-compute"], capture_output=True)
                if img_check.returncode != 0:
                    compute_dir = os.path.join(PROJECT_ROOT, "target_app", "compute_job")
                    print(f"Docker image 'doom-compute' not found. Building from {compute_dir}...")
                    subprocess.run(["docker", "build", "-t", "doom-compute", compute_dir], check=True)

                subprocess.run([
                    "docker", "run", "-d", "--name", "doom-target",
                    "-m", limit_str,
                    "doom-compute"
                ], check=True)

                time.sleep(2)

            elif workload == "flask":
                img_check = subprocess.run(["docker", "image", "inspect", "real-web-server"], capture_output=True)
                if img_check.returncode != 0:
                    target_app_dir = os.path.join(PROJECT_ROOT, "target_app", "flask_app")
                    print(f"Docker image 'real-web-server' not found. Building from {target_app_dir}...")
                    subprocess.run(["docker", "build", "-t", "real-web-server", target_app_dir], check=True)

                subprocess.run([
                    "docker", "run", "-d", "--name", "doom-target",
                    "-m", limit_str,
                    "real-web-server"
                ], check=True)

                time.sleep(2)

                # Retrieve container IP address to bypass host iptables/DNAT port-forwarding issues
                ip_result = subprocess.run(
                    ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", "doom-target"],
                    capture_output=True, text=True, check=True
                )
                container_ip = ip_result.stdout.strip()

                # Fire Apache Bench against the healthy endpoint
                print("Starting asynchronous Apache Bench against healthy endpoint...")
                subprocess.Popen([
                    "ab", "-t", str(duration + 10), "-c", "5", f"http://{container_ip}:5000/matchmake-healthy"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Retrieve PID and cgroup path
            pid_result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Pid}}", "doom-target"],
                capture_output=True, text=True, check=True
            )
            container_pid = pid_result.stdout.strip()

            with open(f"/proc/{container_pid}/cgroup", "r") as f:
                cgroup_content = f.read()

            cgroup_path = ""
            for line in cgroup_content.splitlines():
                if line.startswith("0::"):
                    cgroup_path = line.split("::")[1].strip()
                    break

            if not cgroup_path:
                print("Failed to resolve container cgroup path. Skipping this run...")
                subprocess.run(["docker", "rm", "-f", "doom-target"], capture_output=True)
                continue

            full_cgroup_path = f"/sys/fs/cgroup{cgroup_path}" if cgroup_path.startswith("/") else f"/sys/fs/cgroup/{cgroup_path}"

            # Start eBPF Probe
            ebpfScript = f'tracepoint:kmem:mm_page_alloc /cgroup == cgroupid("{full_cgroup_path}")/ {{ @allocations = count(); }} interval:s:1 {{ print(@allocations); clear(@allocations); }}'
            print("Starting probe...")
            sensor = subprocess.Popen(
                ["stdbuf", "-oL", "bpftrace", "-e", ebpfScript],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )

            # Let sensor attach
            time.sleep(2)

            runData = []
            start_time = time.time()
            aborted = False

            # Read bpftrace stdout asynchronously in a separate thread to avoid blocking issues
            q = queue.Queue()
            t = threading.Thread(target=enqueue_output, args=(sensor.stdout, q))
            t.daemon = True
            t.start()

            try:
                while time.time() - start_time < duration:
                    try:
                        line = q.get(timeout=1.5)
                    except queue.Empty:
                        line = "@allocations: 0"
                    if not line:
                        break

                    if "@allocations" in line:
                        try:
                            velocity = int(line.split(":")[1].strip())
                        except (ValueError, IndexError):
                            continue

                        # Query current memory usage
                        mem_usage = 0
                        try:
                            mem_file = os.path.join(full_cgroup_path, "memory.current")
                            if os.path.exists(mem_file):
                                with open(mem_file, "r") as f:
                                    mem_usage = int(f.read().strip())
                        except Exception:
                            pass

                        # Check if container is still running
                        statusCheck = subprocess.run(
                            ["docker", "inspect", "-f", "{{.State.Running}}", "doom-target"],
                            capture_output=True, text=True
                        )
                        if "false" in statusCheck.stdout:
                            print("Container stopped unexpectedly.")
                            break

                        # Safety check: abort if memory usage exceeds 90% of limit
                        if limit_bytes and mem_usage > 0.90 * limit_bytes:
                            print("Warning: memory > 90% - aborting")
                            aborted = True
                            break

                        pgmajfault, anon, file_mem = get_container_memory_stats(full_cgroup_path)
                        print(f"  mem={mem_usage / (1024*1024):.1f}MB | allocs/s={velocity} | faults={pgmajfault}")
                        runData.append({
                            "Velocity": velocity,
                            "Mem_Usage_Bytes": mem_usage,
                            "Pgmajfault": pgmajfault,
                            "Anon": anon,
                            "File": file_mem
                        })
            except KeyboardInterrupt:
                print("Run interrupted by user.")
                aborted = True
            finally:
                print("Cleaning up...")
                sensor.terminate()
                sensor.wait()
                subprocess.run(["docker", "rm", "-f", "doom-target"], capture_output=True)

            if aborted:
                print("Discarded run")
                continue

            # Record telemetry data if successful
            for data in runData:
                harvested_dataset.append({
                    "Run_ID": run_id_counter,
                    "Workload_Type": workload,
                    "Limit_Bytes": limit_bytes,
                    "Mem_Usage_Bytes": data["Mem_Usage_Bytes"],
                    "Velocity": data["Velocity"],
                    "Pgmajfault": data["Pgmajfault"],
                    "Anon": data["Anon"],
                    "File": data["File"]
                })

            run_id_counter += 1

    # Write harvested data to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Run_ID", "Workload_Type", "Limit_Bytes", "Mem_Usage_Bytes", "Velocity", "Pgmajfault", "Anon", "File"])
        writer.writeheader()
        writer.writerows(harvested_dataset)

    print(f"Exported: {len(harvested_dataset)} rows to {output_path}")
