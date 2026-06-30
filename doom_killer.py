import argparse
import sys
import os

# Ensure the project directory is in the sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.daemon import run_daemon
from src.train import train_model
from src.data_factory import generate_data, harvest_healthy

def main():
    parser = argparse.ArgumentParser(
        description="DOOM-Killer: Predictive OOM Mitigation Engine for Containerized Environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Harvest healthy telemetry across workloads (requires root/sudo)
  sudo python doom_killer.py harvest-healthy --runs 3 --duration 300 --output data/healthyTelemetry.csv

  # Generate crash validation telemetry data (requires root/sudo)
  sudo python doom_killer.py generate-data --runs 50 --output data/crashValidationData.csv

  # Train the predictive model and export to ONNX
  python doom_killer.py train --data data/healthyTelemetry.csv --output models/doom_model.onnx

  # Run the OOM predictive mitigation daemon (requires root/sudo)
  sudo python doom_killer.py run --config config/doom_killer_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")
    
    run_parser = subparsers.add_parser("run", help="Run the OOM mitigation daemon")
    run_parser.add_argument("-c", "--config", help="Path to the config file (json)")
    run_parser.add_argument("-m", "--model", help="Path to the ONNX model file (overrides config)")
    run_parser.add_argument("-t", "--target", default="all", help="Target container name to monitor, or 'all' to monitor all (default: all)")
    
    train_parser = subparsers.add_parser("train", help="Train the anomaly detection model")
    train_parser.add_argument("-d", "--data", default="data/healthyTelemetry.csv", help="Path to input training CSV (default: data/healthyTelemetry.csv)")
    train_parser.add_argument("-o", "--output", default="models/doom_model.onnx", help="Path to output ONNX model (default: models/doom_model.onnx)")
    
    gen_parser = subparsers.add_parser("generate-data", help="Run the simulated telemetry generator")
    gen_parser.add_argument("-o", "--output", default="data/crashValidationData.csv", help="Path to write output training CSV (default: data/crashValidationData.csv)")
    gen_parser.add_argument("-n", "--runs", type=int, default=100, help="Number of simulation runs to execute (default: 100)")

    harvest_parser = subparsers.add_parser("harvest-healthy", help="Harvest healthy telemetry across different workloads")
    harvest_parser.add_argument("-o", "--output", default="data/healthyTelemetry.csv", help="Path to write healthy telemetry CSV (default: data/healthyTelemetry.csv)")
    harvest_parser.add_argument("-w", "--workload", choices=["postgres", "redis", "compute", "flask", "all"], default="all", help="Workload type to harvest (default: all)")
    harvest_parser.add_argument("-d", "--duration", type=int, default=300, help="Duration per run in seconds (default: 300)")
    harvest_parser.add_argument("-n", "--runs", type=int, default=3, help="Number of runs per workload type (default: 3)")
    
    args = parser.parse_args()
    
    if args.command == "run":
        if os.geteuid() != 0:
            print("ERROR: 'run' command must be executed as root (sudo) to load eBPF sensors.")
            sys.exit(1)
        run_daemon(config_path=args.config, model_path_override=args.model, target_override=args.target)
        
    elif args.command == "train":
        success = train_model(data_path=args.data, model_output_path=args.output)
        if not success:
            sys.exit(1)
            
    elif args.command == "generate-data":
        if os.geteuid() != 0:
            print("ERROR: 'generate-data' command must be executed as root (sudo) to run eBPF sensors.")
            sys.exit(1)
        generate_data(output_path=args.output, num_runs=args.runs)

    elif args.command == "harvest-healthy":
        if os.geteuid() != 0:
            print("ERROR: 'harvest-healthy' command must be executed as root (sudo) to run eBPF sensors.")
            sys.exit(1)
        harvest_healthy(output_path=args.output, workload_type=args.workload, duration=args.duration, num_runs=args.runs)

if __name__ == "__main__":
    main()
