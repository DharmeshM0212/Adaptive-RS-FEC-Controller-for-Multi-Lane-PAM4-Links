# experiments/exp_phase6_sweep_controller.py
from __future__ import annotations
import itertools, subprocess, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

BASE_CTRL = {
    "random_seed": 42,
    "io": {
        # We reuse the same M3 metrics you already generated
        "m3_metrics_path": "data/m3_deterministic_m3_metrics.parquet",
        "results_dir": "results",
        "run_name": "m5_ctrl",   # will be suffixed per scenario
    },
    "controller": {
        "parity_levels": [16, 24, 32, 40, 48],
        "default_nsym": 32,
        "energy_cap": 112,       # swept
        "target_sqi": 0.70,
        "softmax_tau": 6.0,      # swept
        "hysteresis_blocks": 2,  # swept
    },
    "simulation": {
        "symbols_per_byte": 4,
        "max_steps": 200,
        "enable_baseline": True,
    },
    "sqi": {
        "snr_db_good": 24.0, "snr_db_bad": 12.0,
        "evm_db_good": 24.0, "evm_db_bad": 10.0,
        "corr_scale": 255.0,
        "w_snr": 0.40, "w_evm": 0.20, "w_corr": 0.30, "w_fail": 0.10,
        "smooth_blocks": 3, "risk_threshold": 0.60,
    },
}

def run(cmd):
    print("[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    else:
        # small, tidy print
        for line in r.stdout.splitlines():
            if line.startswith("[M5]"):
                print(line)

def main():
    project_root = Path(".").resolve()
    cfg_dir = project_root / "configs"; cfg_dir.mkdir(exist_ok=True)
    results_dir = project_root / "results"; results_dir.mkdir(exist_ok=True)

    caps   = [96, 112, 128]
    taus   = [4.0, 6.0, 8.0]
    hysts  = [1, 2, 4]

    rows = []
    for cap, tau, hyst in itertools.product(caps, taus, hysts):
        run_name = f"m5_ctrl_cap{cap}_tau{int(tau)}_hy{hyst}"
        ctrl_cfg = BASE_CTRL.copy()
        ctrl_cfg = yaml.safe_load(yaml.safe_dump(ctrl_cfg))  # deep copy via yaml
        ctrl_cfg["io"]["run_name"] = run_name
        ctrl_cfg["controller"]["energy_cap"] = int(cap)
        ctrl_cfg["controller"]["softmax_tau"] = float(tau)
        ctrl_cfg["controller"]["hysteresis_blocks"] = int(hyst)

        # write scenario config
        cfg_path = cfg_dir / "ctrl.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(ctrl_cfg, f, sort_keys=False)

        # run controller simulation (produces timeline parquet)
        run([sys.executable, "-m", "experiments.exp_phase_5_sim_controller"])

        # read timeline + compute summary
        tl_path = results_dir / f"{run_name}_timeline.parquet"
        if not tl_path.exists():
            print(f"[SWEEP] Timeline missing for {run_name}, skipping.")
            continue
        df = pd.read_parquet(tl_path)
        nlanes = 4 if "nsym_alloc" not in df.columns else len(df["nsym_alloc"].iloc[0])

        bler_adapt = (df["fails_adapt"].mean() / max(1, nlanes)) if not df.empty else np.nan
        bler_base  = (df["fails_base"].mean() / max(1, nlanes)) if ("fails_base" in df.columns) and (not df.empty) else np.nan
        energy     = df["energy_used"].mean() if not df.empty else np.nan

        rows.append(dict(
            run_name=run_name, cap=cap, tau=tau, hyst=hyst,
            bler_adapt=bler_adapt, bler_base=bler_base,
            delta= (bler_base - bler_adapt) if (pd.notna(bler_base) and pd.notna(bler_adapt)) else np.nan,
            delta_pct= 100.0*(bler_base - bler_adapt)/bler_base if (pd.notna(bler_base) and bler_base>0) else np.nan,
            energy=energy,
        ))

    summ = pd.DataFrame(rows).sort_values(["cap","tau","hyst"]).reset_index(drop=True)
    out_csv = results_dir / "m6_sweep_summary.csv"
    summ.to_csv(out_csv, index=False)
    print(f"[SWEEP] Saved summary: {out_csv}")
    print(summ.head(10))

    # --- Simple heatmaps: Δ% vs baseline per (cap, tau) @ hyst=2 ---
    try:
        pivot = summ[summ["hyst"]==2].pivot(index="cap", columns="tau", values="delta_pct")
        plt.figure(figsize=(6,4))
        plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.xticks(range(pivot.shape[1]), pivot.columns)
        plt.yticks(range(pivot.shape[0]), pivot.index)
        plt.colorbar(label="Δ BLER vs baseline (%)   (positive = better)")
        plt.title("Adaptive Gain vs Baseline by Energy Cap & Tau (hyst=2)")
        plt.xlabel("Softmax tau")
        plt.ylabel("Energy cap (sum nsym)")
        fig = results_dir / "m6_heatmap_delta_pct.png"
        plt.tight_layout(); plt.savefig(fig, dpi=150)
        print(f"[SWEEP] Saved heatmap: {fig}")
    except Exception as e:
        print(f"[SWEEP] Heatmap skipped: {e}")

if __name__ == "__main__":
    main()
