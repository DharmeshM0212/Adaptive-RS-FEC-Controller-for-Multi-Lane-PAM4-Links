# experiments/exp_phase5_sim_controller.py
from __future__ import annotations
import numpy as np, pandas as pd, yaml
from pathlib import Path
from typing import Optional, List
from controllers.allocator import allocate_nsym_from_sqi
from scripts.sqi import compute_sqi  # reuse SQI from M4

DEFAULT_CFG = {
    "random_seed": 42,
    "io": {
        "m3_metrics_path": "data/m3_deterministic_m3_metrics.parquet",
        "results_dir": "results",
        "run_name": "m5_ctrl",
    },
    "controller": {
        "parity_levels": [16, 24, 32, 40, 48],
        "default_nsym": 32,
        "energy_cap": 112,
        "target_sqi": 0.70,
        "softmax_tau": 6.0,
        "hysteresis_blocks": 2,
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

def load_cfg(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[M5] WARNING: {p} not found. Using defaults.")
        return DEFAULT_CFG
    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    # shallow merge
    merged = DEFAULT_CFG.copy()
    for k, v in cfg.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            mv = merged[k].copy(); mv.update(v); merged[k] = mv
        else:
            merged[k] = v
    return merged

def ser_to_pbyte(ser: float, symbols_per_byte: int = 4) -> float:
    ser = float(np.clip(ser, 0.0, 1.0))
    return 1.0 - (1.0 - ser)**int(symbols_per_byte)

def simulate_rs_block(p_byte: float, nsym: int, n_bytes: int = 255, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng()
    errors = rng.binomial(n=n_bytes, p=np.clip(p_byte, 0.0, 1.0))
    t = int(nsym)//2
    return int(errors > t), int(errors)

def main():
    cfg = load_cfg("configs/ctrl.yaml")
    m3_path = Path(cfg["io"]["m3_metrics_path"])
    results_dir = Path(cfg["io"]["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["io"]["run_name"]

    if not m3_path.exists():
        raise FileNotFoundError(f"[M5] M3 metrics not found at {m3_path}")

    # Load M3 + ensure block axis exists
    df = pd.read_parquet(m3_path)
    if df.empty:
        raise ValueError("[M5] M3 parquet is empty.")
    if "block" not in df.columns:
        df = df.sort_values(["lane"]).reset_index(drop=True)
        df["block"] = df.groupby("lane").cumcount()

    # Compute SQI
    df_sqi = compute_sqi(df, cfg)  # adds 'sqi'
    df_sqi = df_sqi.sort_values(["block","lane"]).reset_index(drop=True)

    lanes = sorted(df_sqi["lane"].unique())
    times = sorted(df_sqi["block"].unique())
    max_steps = int(cfg["simulation"]["max_steps"])
    if max_steps > 0:
        times = times[:max_steps]

    levs: List[int] = list(cfg["controller"]["parity_levels"])
    cap = int(cfg["controller"]["energy_cap"])
    nsym_default = int(cfg["controller"]["default_nsym"])
    target_sqi = float(cfg["controller"]["target_sqi"])
    tau = float(cfg["controller"]["softmax_tau"])
    hyst = int(cfg["controller"]["hysteresis_blocks"])
    sym_per_byte = int(cfg["simulation"]["symbols_per_byte"])
    use_baseline = bool(cfg["simulation"]["enable_baseline"])

    rng = np.random.default_rng(int(cfg["random_seed"]))

    history: List[dict] = []
    last_alloc = None
    hysteresis_count = 0

    print(f"[M5] Running controller on {len(times)} blocks, lanes={len(lanes)}, cap={cap}")
    for t in times:
        frame = df_sqi[df_sqi["block"] == t].set_index("lane")

        # Reindex to ensure *all lanes* exist at each t; ffill from last known
        # If a lane has no past row, fill with neutral defaults.
        if set(frame.index) != set(lanes):
            missing = [ln for ln in lanes if ln not in frame.index]
            for ln in missing:
                prev = df_sqi[(df_sqi["lane"] == ln) & (df_sqi["block"] < t)]
                if prev.empty:
                    filler = pd.Series({"block": t, "sqi": 0.85, "ser": 0.0}, name=ln)
                else:
                    last = prev.iloc[-1]
                    filler = pd.Series({"block": t, "sqi": float(last["sqi"]), "ser": float(last["ser"])}, name=ln)
                frame.loc[ln] = filler
        frame = frame.sort_index()

        sqi_vec = frame["sqi"].to_numpy(dtype=float)
        ser_vec = frame["ser"].to_numpy(dtype=float)
        # --- Optional time-varying SQI events to force reallocations ---
        evts = cfg.get("events", [])
        if evts:
            for e in evts:
                if int(e.get("lane", -1)) in range(len(sqi_vec)):
                    if int(e["start_block"]) <= int(t) < int(e["end_block"]):
                        sqi_vec[int(e["lane"])] = float(np.clip(
                            sqi_vec[int(e["lane"])] + float(e["sqi_delta"]), 0.0, 1.0
                        ))


        # Hysteresis
        if hysteresis_count > 0 and last_alloc is not None:
            alloc = last_alloc.copy()
            hysteresis_count -= 1
        else:
            alloc = allocate_nsym_from_sqi(
                sqi_vec, parity_levels=levs, energy_cap=cap, default_nsym=nsym_default,
                target_sqi=target_sqi, tau=tau
            )
            if last_alloc is not None and np.any(alloc != last_alloc):
                hysteresis_count = max(0, hyst - 1)
            last_alloc = alloc.copy()

        pbyte_vec = np.array([ser_to_pbyte(s, symbols_per_byte=sym_per_byte) for s in ser_vec], dtype=float)

        # Simulate outcomes
        fails_adapt = []; errs_adapt = []
        fails_base  = []; errs_base  = []
        for i in range(len(lanes)):
            f_a, e_a = simulate_rs_block(pbyte_vec[i], nsym=int(alloc[i]), rng=rng)
            fails_adapt.append(f_a); errs_adapt.append(e_a)
            if use_baseline:
                f_b, e_b = simulate_rs_block(pbyte_vec[i], nsym=nsym_default, rng=rng)
                fails_base.append(f_b); errs_base.append(e_b)

        history.append(dict(
            block=int(t),
            nsym_alloc=list(map(int, alloc)),
            sqi=list(map(float, sqi_vec)),
            ser=list(map(float, ser_vec)),
            fails_adapt=int(np.sum(fails_adapt)),
            fails_base=int(np.sum(fails_base)) if use_baseline else None,
            energy_used=int(np.sum(alloc)),
        ))

    # Build output DataFrame with guaranteed columns
    out_cols = ["block","nsym_alloc","sqi","ser","fails_adapt","fails_base","energy_used"]
    out = pd.DataFrame(history, columns=out_cols)

    out_path = results_dir / f"{run_name}_timeline.parquet"
    out.to_parquet(out_path, index=False)

    if out.empty:
        print(f"[M5] WARNING: No timeline rows produced. Check your M3 file and blocks_per_run.")
        return

    bler_adapt = out["fails_adapt"].mean() / max(1, len(lanes))
    bler_base  = (out["fails_base"].mean() / max(1, len(lanes))) if use_baseline else np.nan

    print(f"[M5] Saved timeline: {out_path}")
    print(f"[M5] Avg BLER (adaptive) = {bler_adapt:.4f}")
    if use_baseline:
        print(f"[M5] Avg BLER (baseline nsym={nsym_default}) = {bler_base:.4f}")
    print(f"[M5] Avg energy used (sum nsym) = {out['energy_used'].mean():.1f}  (cap {cap})")

if __name__ == "__main__":
    main()
