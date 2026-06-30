import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

"""
    Loads healthy telemetry dataset, processes it, performs leave-one-workload-out
    validation, trains a final IsolationForest model, and exports it to ONNX format.
"""

def train_model(data_path, model_output_path):
    print(f"Loading dataset: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found")
        return False
        
    df = pd.read_csv(data_path)
    
    # Sort each run
    if "TTO_Seconds" in df.columns:
        df = df.sort_values(by=["Run_ID", "TTO_Seconds"], ascending=[True, False]).reset_index(drop=True)
    else:
        df = df.sort_values(by=["Run_ID"]).reset_index(drop=True)
        
    print("Engineering features...")
    # Remaining Memory Ratio (headroom)
    df["Remaining_Memory_Ratio"] = (df["Limit_Bytes"] - df["Mem_Usage_Bytes"]) / df["Limit_Bytes"]
    
    # Relative Velocity: Fraction of limit allocated per second (1 page = 4096 bytes)
    df["Relative_Velocity"] = (df["Velocity"] * 4096.0) / df["Limit_Bytes"]
    
    # Relative Acceleration
    df["Relative_Acceleration"] = df.groupby("Run_ID")["Relative_Velocity"].diff().fillna(0)
    
    # Major page fault rate (diff per second)
    df["Majfault_Rate"] = df.groupby("Run_ID")["Pgmajfault"].diff().fillna(0)

    # Cache-to-RSS ratio (file / anon)
    df["Cache_to_RSS"] = df["File"] / df["Anon"].clip(lower=1)

    # Rolling averages (window=3)
    df["Relative_Velocity_Roll_3"] = df.groupby("Run_ID")["Relative_Velocity"].transform(lambda x: x.rolling(3).mean())
    df["Relative_Acceleration_Roll_3"] = df.groupby("Run_ID")["Relative_Acceleration"].transform(lambda x: x.rolling(3).mean())
    df["Majfault_Rate_Roll_3"] = df.groupby("Run_ID")["Majfault_Rate"].transform(lambda x: x.rolling(3).mean())
    df["Cache_to_RSS_Roll_3"] = df.groupby("Run_ID")["Cache_to_RSS"].transform(lambda x: x.rolling(3).mean())
    
    # Fill nan
    df["Relative_Velocity_Roll_3"] = df["Relative_Velocity_Roll_3"].fillna(df["Relative_Velocity"])
    df["Relative_Acceleration_Roll_3"] = df["Relative_Acceleration_Roll_3"].fillna(0)
    df["Majfault_Rate_Roll_3"] = df["Majfault_Rate_Roll_3"].fillna(df["Majfault_Rate"])
    df["Cache_to_RSS_Roll_3"] = df["Cache_to_RSS_Roll_3"].fillna(df["Cache_to_RSS"])
    
    features = [
        "Remaining_Memory_Ratio",
        "Relative_Velocity",
        "Relative_Acceleration",
        "Relative_Velocity_Roll_3",
        "Relative_Acceleration_Roll_3",
        "Majfault_Rate",
        "Cache_to_RSS",
        "Majfault_Rate_Roll_3",
        "Cache_to_RSS_Roll_3"
    ]
    
    print(f"Dataset size: {len(df)} samples")

    if "Workload_Type" not in df.columns:
        df["Workload_Type"] = "unknown"

    workloads = df["Workload_Type"].unique()

    crash_df = None
    crash_path = os.path.join(os.path.dirname(data_path), "crashValidationData.csv")
    if os.path.exists(crash_path):
        try:
            cdf = pd.read_csv(crash_path)
            cdf["Remaining_Memory_Ratio"] = (cdf["Limit_Bytes"] - cdf["Mem_Usage_Bytes"]) / cdf["Limit_Bytes"]
            cdf["Relative_Velocity"] = (cdf["Velocity"] * 4096.0) / cdf["Limit_Bytes"]
            cdf["Relative_Acceleration"] = cdf.groupby("Run_ID")["Relative_Velocity"].diff().fillna(0)
            cdf["Majfault_Rate"] = cdf.groupby("Run_ID")["Pgmajfault"].diff().fillna(0)
            cdf["Cache_to_RSS"] = cdf["File"] / cdf["Anon"].clip(lower=1)
            cdf["Relative_Velocity_Roll_3"] = cdf.groupby("Run_ID")["Relative_Velocity"].transform(lambda x: x.rolling(3).mean())
            cdf["Relative_Acceleration_Roll_3"] = cdf.groupby("Run_ID")["Relative_Acceleration"].transform(lambda x: x.rolling(3).mean())
            cdf["Majfault_Rate_Roll_3"] = cdf.groupby("Run_ID")["Majfault_Rate"].transform(lambda x: x.rolling(3).mean())
            cdf["Cache_to_RSS_Roll_3"] = cdf.groupby("Run_ID")["Cache_to_RSS"].transform(lambda x: x.rolling(3).mean())
            cdf["Relative_Velocity_Roll_3"] = cdf["Relative_Velocity_Roll_3"].fillna(cdf["Relative_Velocity"])
            cdf["Relative_Acceleration_Roll_3"] = cdf["Relative_Acceleration_Roll_3"].fillna(0)
            cdf["Majfault_Rate_Roll_3"] = cdf["Majfault_Rate_Roll_3"].fillna(cdf["Majfault_Rate"])
            cdf["Cache_to_RSS_Roll_3"] = cdf["Cache_to_RSS_Roll_3"].fillna(cdf["Cache_to_RSS"])
            crash_df = cdf
        except Exception:
            pass

    print("Running cross-validation splits...")
    for val_workload in workloads:
        train_idx = df["Workload_Type"] != val_workload
        val_idx = df["Workload_Type"] == val_workload

        if not train_idx.any() or not val_idx.any():
            continue

        X_train_val = df.loc[train_idx, features].astype(np.float32)
        X_val_healthy = df.loc[val_idx, features].astype(np.float32)

        val_model = IsolationForest(contamination=0.01, random_state=42)
        val_model.fit(X_train_val)

        # anomaly_score = -decision_function is positive for anomalies and negative for inliers.
        healthy_scores = -val_model.decision_function(X_val_healthy)
        
        print(f"  Holdout: {val_workload} | samples={len(X_val_healthy)} | healthy_score_mean={healthy_scores.mean():.4f} max={healthy_scores.max():.4f}")

        if val_workload == "flask" and crash_df is not None:
            X_val_crash = crash_df[features].astype(np.float32)
            crash_scores = -val_model.decision_function(X_val_crash)
            
            for tto_range, name in [((0, 2), "Imminent (0-2s)"), ((3, 5), "Approaching (3-5s)")]:
                mask = (crash_df["TTO_Seconds"] >= tto_range[0]) & (crash_df["TTO_Seconds"] <= tto_range[1])
                if mask.any():
                    range_scores = crash_scores[mask]
                    print(f"    Flask holdout validation -> {name}: mean={range_scores.mean():.4f} max={range_scores.max():.4f}")

    print("Training model...")
    X_all = df[features].astype(np.float32)
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_all)

    print("Exporting ONNX...")
    initial_types = [('float_input', FloatTensorType([None, len(features)]))]
    
    onnx_model = to_onnx(model, initial_types=initial_types, target_opset={'': 12, 'ai.onnx.ml': 3})

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"Saved: {model_output_path}")
    return True
