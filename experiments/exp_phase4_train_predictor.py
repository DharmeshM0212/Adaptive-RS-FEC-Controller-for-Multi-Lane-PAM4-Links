# experiments/exp_phase4_train_predictor.py
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# pip install scikit-learn matplotlib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt


DEFAULT_CFG = {
    "io": {
        "dataset_out": "data/m4_sqi_dataset.parquet",
        "results_dir": "results",
        "run_name": "m4_predictor",
    },
    "sqi": {
        "risk_threshold": 0.60,
    },
    "dataset": {
        "train_frac": 0.7,
    },
    "model": {
        "type": "gbr",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "random_state": 42,
        },
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
    # shallow merge with defaults
    merged = DEFAULT_CFG.copy()
    for k, v in cfg.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            mv = merged[k].copy()
            mv.update(v)
            merged[k] = mv
        else:
            merged[k] = v
    return merged


def chronological_split_mask(n_rows: int, frac: float) -> np.ndarray:
    split = max(1, min(n_rows - 1, int(n_rows * frac)))
    m = np.zeros(n_rows, dtype=bool)
    m[:split] = True
    return m


def main():
    cfg = load_cfg_safe("configs/m4.yaml")

    ds_path = Path(cfg["io"]["dataset_out"])
    results_dir = Path(cfg["io"]["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["io"]["run_name"]

    if not ds_path.exists():
        raise FileNotFoundError(f"[M4] Dataset not found at {ds_path}. Run exp_phase4_build_dataset first.")

    ds = pd.read_parquet(ds_path)
    if ds.empty:
        raise ValueError("[M4] Loaded dataset is empty. Rebuild with exp_phase4_build_dataset (increase blocks/window).")

    # Features = the flattened time series columns created by the builder
    feature_cols = [c for c in ds.columns if any(
        c.startswith(p) for p in ("sqi_", "sqi_raw_", "snr_db_", "evm_db_", "rs_corr_mean_", "rs_fail_cw_", "ser_")
    )]
    if not feature_cols:
        raise ValueError("[M4] No feature columns found. Check dataset build step / column prefixes.")

    X = ds[feature_cols].to_numpy(dtype=np.float32)
    y = ds["y"].to_numpy()

    print(f"[M4] Loaded dataset: rows={X.shape[0]}, dims={X.shape[1]}, features={len(feature_cols)}")

    # --- Train/test split (robust to degenerate tY) ---
    frac = float(cfg["dataset"]["train_frac"])
    if "tY" in ds.columns and ds["tY"].nunique() >= 5:
        tY_sorted = np.sort(ds["tY"].unique())
        split_idx = max(1, int(len(tY_sorted) * frac))
        tY_train = set(tY_sorted[:split_idx])
        is_train = ds["tY"].isin(tY_train).to_numpy()
        if is_train.sum() == 0 or is_train.sum() == len(is_train):
            print("[M4] WARNING: time split degenerate; falling back to chronological split by rows.")
            is_train = chronological_split_mask(len(ds), frac)
    else:
        print("[M4] INFO: 'tY' missing or not varied; using chronological split by rows.")
        is_train = chronological_split_mask(len(ds), frac)

    Xtr, Ytr = X[is_train], y[is_train]
    Xte, Yte = X[~is_train], y[~is_train]
    print(f"[M4] Split: train={Xtr.shape[0]} samples, test={Xte.shape[0]} samples")

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        raise ValueError("[M4] Split produced empty train/test. Adjust train_frac or dataset window/horizon.")

    # --- Model: Gradient Boosting Regressor on future SQI ---
    mdl_cfg = cfg.get("model", DEFAULT_CFG["model"])
    params = mdl_cfg.get("params", DEFAULT_CFG["model"]["params"])
    model = GradientBoostingRegressor(**params)
    model.fit(Xtr, Ytr)

    # --- Predict ---
    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)

    # --- Regression quality ---
    mae_tr = mean_absolute_error(Ytr, yhat_tr)
    mae_te = mean_absolute_error(Yte, yhat_te)
    print(f"[M4] Regressor MAE: train={mae_tr:.4f}  test={mae_te:.4f}")

    # --- Early-risk classification derived from predicted SQI ---
    thr = float(cfg["sqi"]["risk_threshold"])
    risk_true_te = (Yte < thr).astype(int)
    risk_score_te = 1.0 - yhat_te  # lower predicted SQI → higher risk

    if len(np.unique(risk_true_te)) >= 2:
        auc_te = roc_auc_score(risk_true_te, risk_score_te)
        print(f"[M4] Risk ROC-AUC (test) = {auc_te:.3f}")
        risk_pred_te = (yhat_te < thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(risk_true_te, risk_pred_te, average="binary", zero_division=0)
        print(f"[M4] Test Risk@thr: precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")

        # Plot ROC
        fpr, tpr, _ = roc_curve(risk_true_te, risk_score_te)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc_te:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Lane-Risk ROC (test)")
        plt.legend()
        fig_path = results_dir / f"{run_name}_roc.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"[M4] Saved ROC: {fig_path}")
    else:
        print("[M4] Note: test set has a single class for risk; ROC/AUC not defined at this split.")

if __name__ == "__main__":
    main()
