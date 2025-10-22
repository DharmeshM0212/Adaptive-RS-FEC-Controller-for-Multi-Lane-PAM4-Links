# scripts/sqi.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))

def _lin01(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # map [lo..hi] → [0..1]; clamp outside
    if hi == lo:
        return np.ones_like(x) * 0.5
    return _clamp01((x - lo) / (hi - lo))

def compute_sqi(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add SQI + risk label to per-block metrics dataframe from M3."""
    df = df.copy()

    # Pull knobs
    snr_good = float(cfg["sqi"]["snr_db_good"])
    snr_bad  = float(cfg["sqi"]["snr_db_bad"])
    evm_good = float(cfg["sqi"]["evm_db_good"])
    evm_bad  = float(cfg["sqi"]["evm_db_bad"])
    w_snr    = float(cfg["sqi"]["w_snr"])
    w_evm    = float(cfg["sqi"]["w_evm"])
    w_corr   = float(cfg["sqi"]["w_corr"])
    w_fail   = float(cfg["sqi"]["w_fail"])
    corr_scale = float(cfg["sqi"]["corr_scale"])
    smooth_n   = int(cfg["sqi"]["smooth_blocks"])
    thr        = float(cfg["sqi"]["risk_threshold"])

    # Normalize “good is high”
    snr_score = _lin01(df["snr_db"].to_numpy(), lo=snr_bad, hi=snr_good)
    evm_score = _lin01(df["evm_db"].to_numpy(), lo=evm_bad, hi=evm_good)

    # RS correction utilization (0..1), good is LOW → score = 1 - util
    corr_util = np.minimum(1.0, df["rs_corr_mean"].fillna(0.0).to_numpy() / max(1.0, corr_scale))
    corr_score = 1.0 - corr_util

    # Fail score: if this block failed, score=0 else 1
    fail_score = 1.0 - np.minimum(1.0, df["rs_fail_cw"].fillna(0.0).to_numpy())

    # Weighted SQI in [0..1]
    sqi = (w_snr * snr_score +
           w_evm * evm_score +
           w_corr * corr_score +
           w_fail * fail_score)

    # Optional rolling smoothing per lane (causal)
    df["sqi_raw"] = sqi
    df = df.sort_values(["lane", "block"])
    if smooth_n > 1:
        df["sqi"] = df.groupby("lane")["sqi_raw"].transform(lambda s: s.rolling(smooth_n, min_periods=1).mean())
    else:
        df["sqi"] = df["sqi_raw"]

    # Risk label
    df["risk_label"] = (df["sqi"] < thr).astype(int)

    return df

def build_time_series(df: pd.DataFrame, window: int, horizon: int,
                      feature_cols: list[str], target: str = "sqi") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Turn per-lane block metrics into (X, y) time-series samples with window W predicting t+H.
    X shape: [samples, window, features]
    y shape: [samples] (sqi or risk_label)
    Returns X, y, and an index dataframe (lane, t0, tY)
    """
    rows = []
    X_list, y_list = [], []
    df = df.sort_values(["lane", "block"])
    lanes = sorted(df["lane"].unique())
    for lane in lanes:
        d = df[df["lane"] == lane].reset_index(drop=True)
        T = len(d)
        for t0 in range(0, T - window - horizon + 1):
            tX = slice(t0, t0 + window)
            tY = t0 + window - 1 + horizon
            Xi = d.loc[tX, feature_cols].to_numpy(dtype=np.float32)
            if target == "sqi":
                yi = float(d.loc[tY, "sqi"])
            elif target == "risk":
                yi = int(d.loc[tY, "risk_label"])
            else:
                raise ValueError("target must be 'sqi' or 'risk'")
            X_list.append(Xi)
            y_list.append(yi)
            rows.append(dict(lane=lane, t0=t0, tY=int(d.loc[tY, "block"])))
    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, window, len(feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32 if target == "sqi" else np.int32)
    meta = pd.DataFrame(rows)
    return X, y, meta
