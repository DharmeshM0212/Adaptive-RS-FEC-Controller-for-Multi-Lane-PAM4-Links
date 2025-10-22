# experiments/exp_phase5_plot_results.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json

def _numeric_block_axis(df: pd.DataFrame) -> np.ndarray:
    """Return a numeric time axis. If 'block' is missing/non-numeric/constant, use row index."""
    if "block" in df.columns:
        x = pd.to_numeric(df["block"], errors="coerce").to_numpy()
        if np.isfinite(x).all() and (np.nanmax(x) != np.nanmin(x)):
            return x
    # fallback
    return np.arange(len(df), dtype=float)

def main():
    cfg_path = Path("configs/ctrl.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    else:
        cfg = {}
    results_dir = Path(cfg.get("io", {}).get("results_dir", "results"))
    run_name    = cfg.get("io", {}).get("run_name", "m5_ctrl")
    cap         = int(cfg.get("controller", {}).get("energy_cap", 112))
    tl_path     = results_dir / f"{run_name}_timeline.parquet"
    if not tl_path.exists():
        raise FileNotFoundError(f"Timeline parquet not found: {tl_path}")

    df = pd.read_parquet(tl_path).copy()
    # Expand per-lane parity column
    max_len = int(df["nsym_alloc"].map(len).max())
    lane_cols = []
    for i in range(max_len):
        col = f"nsym_lane{i}"
        lane_cols.append(col)
        df[col] = df["nsym_alloc"].map(lambda v: v[i] if i < len(v) else np.nan)

    # BLER columns
    nlanes = len(lane_cols)
    df["bler_adapt"] = df["fails_adapt"] / max(1, nlanes)
    has_base = "fails_base" in df.columns and not df["fails_base"].isna().all()
    if has_base:
        df["bler_base"] = df["fails_base"] / max(1, nlanes)

    # Clean time axis
    x = _numeric_block_axis(df)

    # ========== (1) Parity per lane vs time (readable) ==========
    plt.figure(figsize=(11, 4))
    jitter = np.linspace(-0.15, 0.15, nlanes)  # spread lanes slightly on x
    for i, col in enumerate(lane_cols):
        plt.plot(x + jitter[i], df[col], marker="o", markersize=2, linewidth=1, label=f"Lane {i}", alpha=0.9)
    plt.axhline(cap / max(1, nlanes), linestyle="--", linewidth=1, label="Avg budget/lane")
    plt.title("Adaptive RS Parity Allocation per Lane (nsym)")
    plt.xlabel("Block")
    plt.ylabel("Parity bytes (nsym)")
    plt.legend(ncol=min(5, nlanes), fontsize=8)
    fig1 = results_dir / f"{run_name}_parity_per_lane.png"
    plt.tight_layout(); plt.savefig(fig1, dpi=150)

    # ========== (2) Lane×time parity heatmap ==========
    # makes variation obvious even if lines overlap
    H = df[lane_cols].to_numpy().T  # [lanes, time]
    plt.figure(figsize=(11, 3.6))
    plt.imshow(H, aspect="auto", interpolation="nearest", origin="lower")
    plt.colorbar(label="nsym")
    plt.yticks(range(nlanes), [f"Lane {i}" for i in range(nlanes)])
    plt.xticks(np.linspace(0, len(x)-1, min(10, len(x))), [f"{int(v)}" for v in np.linspace(x.min(), x.max(), min(10, len(x)))])
    plt.title("Parity Allocation Heatmap (lanes × time)")
    fig1b = results_dir / f"{run_name}_parity_heatmap.png"
    plt.tight_layout(); plt.savefig(fig1b, dpi=150)

    # ========== (3) BLER per block + cumulative mean ==========
    plt.figure(figsize=(11, 4))
    plt.plot(x, df["bler_adapt"], alpha=0.35, label="BLER (adaptive)")
    plt.plot(x, df["bler_adapt"].expanding().mean(), linewidth=2, label="Mean BLER (adaptive)")
    if has_base:
        plt.plot(x, df["bler_base"], alpha=0.35, label="BLER (baseline)")
        plt.plot(x, df["bler_base"].expanding().mean(), linewidth=2, label="Mean BLER (baseline)")
    plt.title("Block Error Rate (per block) and Cumulative Mean")
    plt.xlabel("Block"); plt.ylabel("BLER")
    plt.legend()
    fig2 = results_dir / f"{run_name}_bler_timeseries.png"
    plt.tight_layout(); plt.savefig(fig2, dpi=150)

    # ========== (4) Energy used vs cap ==========
    plt.figure(figsize=(11, 3))
    plt.plot(x, df["energy_used"], linewidth=2, label="Energy used (Σ nsym)")
    plt.axhline(cap, linestyle="--", linewidth=1, label=f"Cap ({cap})")
    plt.title("Total Parity Budget Over Time")
    plt.xlabel("Block"); plt.ylabel("Sum of nsym across lanes")
    plt.legend()
    fig3 = results_dir / f"{run_name}_energy_vs_cap.png"
    plt.tight_layout(); plt.savefig(fig3, dpi=150)

    # ========== (5) Distributions to prove it's not flat ==========
    # per-lane nsym hist + BLER hist
    plt.figure(figsize=(11, 3.6))
    for i, col in enumerate(lane_cols):
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if len(vals):
            plt.hist(vals, bins=np.arange(min(vals), max(vals)+4, 4), alpha=0.5, label=f"Lane {i}")
    plt.xlabel("nsym"); plt.ylabel("count"); plt.title("Parity Allocation Distribution")
    plt.legend(fontsize=8)
    fig4 = results_dir / f"{run_name}_parity_hist.png"
    plt.tight_layout(); plt.savefig(fig4, dpi=150)

    plt.figure(figsize=(11, 3.6))
    plt.hist(df["bler_adapt"], bins=20, alpha=0.7, label="adaptive")
    if has_base:
        plt.hist(df["bler_base"], bins=20, alpha=0.7, label="baseline")
    plt.xlabel("BLER per block"); plt.ylabel("count"); plt.title("BLER Distribution")
    plt.legend()
    fig5 = results_dir / f"{run_name}_bler_hist.png"
    plt.tight_layout(); plt.savefig(fig5, dpi=150)

    # ========== (6) Summary CSV ==========
    p95 = lambda s: float(np.nanpercentile(s, 95)) if len(s) else np.nan
    summary = {
        "bler_adapt_mean": float(df["bler_adapt"].mean()),
        "bler_adapt_p95":  p95(df["bler_adapt"]),
        "energy_mean":     float(df["energy_used"].mean()),
        "energy_cap":      float(cap),
    }
    if has_base:
        summary.update({
            "bler_base_mean": float(df["bler_base"].mean()),
            "bler_base_p95":  p95(df["bler_base"]),
            "delta_mean":     float(df["bler_base"].mean() - df["bler_adapt"].mean()),
            "delta_mean_pct": float(100.0 * (df["bler_base"].mean() - df["bler_adapt"].mean()) / max(1e-9, df["bler_base"].mean())),
        })
    out_csv = results_dir / f"{run_name}_summary.csv"
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    print("[PLOT] Wrote:")
    for p in [fig1, fig1b, fig2, fig3, fig4, fig5, out_csv]:
        print(" ", p)

if __name__ == "__main__":
    main()
