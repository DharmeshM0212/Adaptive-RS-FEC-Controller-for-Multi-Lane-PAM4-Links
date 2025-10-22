# experiments/exp_phase4_build_dataset.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from scripts.sqi import compute_sqi, build_time_series

FEATURES = [
    "sqi", "sqi_raw", "snr_db", "evm_db", "rs_corr_mean", "rs_fail_cw", "ser",
]

DEFAULT_CFG = {
    "io": {
        "m3_metrics_path": "data/m3_deterministic_m3_metrics.parquet",
        "dataset_out":     "data/m4_sqi_dataset.parquet",
        "results_dir":     "results",
        "run_name":        "m4_predictor",
    },
    "sqi": {
        "snr_db_good": 24.0, "snr_db_bad": 12.0,
        "evm_db_good": 24.0, "evm_db_bad": 10.0,
        "corr_scale": 255.0,
        "w_snr": 0.40, "w_evm": 0.20, "w_corr": 0.30, "w_fail": 0.10,
        "smooth_blocks": 3, "risk_threshold": 0.60,
    },
    "dataset": {
        "window_len": 8, "horizon": 4, "train_frac": 0.7, "target": "sqi",
    },
}

def load_cfg_safe(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[M4] WARNING: {p} not found. Using defaults.")
        return DEFAULT_CFG
    try:
        with p.open("r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[M4] WARNING: failed to read {p}: {e}. Using defaults.")
        return DEFAULT_CFG
    if not cfg:
        print(f"[M4] WARNING: {p} is empty. Using defaults.")
        return DEFAULT_CFG
    # shallow merge
    merged = DEFAULT_CFG.copy()
    for k, v in cfg.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            mv = merged[k].copy()
            mv.update(v)
            merged[k] = mv
        else:
            merged[k] = v
    return merged

def main():
    cfg = load_cfg_safe("configs/m4.yaml")

    in_path  = Path(cfg["io"]["m3_metrics_path"])
    out_path = Path(cfg["io"]["dataset_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"[M4] Could not find M3 metrics parquet at '{in_path}'. "
            f"Run Phase-3 first or set io.m3_metrics_path in configs/m4.yaml."
        )

    df = pd.read_parquet(in_path)

    # Ensure a block index exists (time axis per lane)
    if "block" not in df.columns:
        df = df.sort_values(["lane"]).reset_index(drop=True)
        df["block"] = df.groupby("lane").cumcount()

    # Compute SQI + risk labels
    df_sqi = compute_sqi(df, cfg)

    # Build time-series dataset
    W = int(cfg["dataset"]["window_len"])
    H = int(cfg["dataset"]["horizon"])
    target = str(cfg["dataset"]["target"])
    X, y, meta = build_time_series(df_sqi, window=W, horizon=H, feature_cols=FEATURES, target=target)

    # Flatten [N, W, F] → [N, W*F] for tabular ML baselines
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F) if N else np.zeros((0, 0), dtype=np.float32)

    # Build column names from actual T and F to avoid mismatch
    # Try to use FEATURES if it matches F; otherwise fall back to generic names.
    if F == len(FEATURES):
        step_names = FEATURES
    else:
        step_names = [f"f{j}" for j in range(F)]

    colnames = [f"{c}_t{-k}" for k in range(T, 0, -1) for c in step_names]

    # Now shapes will always match
    ds = pd.DataFrame(X_flat, columns=colnames)
    ds["y"] = y
    ds = pd.concat([meta, ds], axis=1)
    ds.to_parquet(out_path, index=False)

    print(f"[M4] Built dataset: {out_path}")
    print(f"  samples={N}, window(T)={T}, features/step(F)={F}, total_cols={T*F}")
    if N:
        print(ds.head(min(3, len(ds))))

if __name__ == "__main__":
    main()
