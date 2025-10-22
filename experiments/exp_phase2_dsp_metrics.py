# experiments/exp_phase2_dsp_metrics.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from scripts.dsp_equalizer import dsp_chain, hard_decision_pam4
from scripts.metrics import compute_evm_db, estimate_snr_db, eye_metrics_approx

def ensure_dirs(cfg):
    Path(cfg["io"]["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["io"]["results_dir"]).mkdir(parents=True, exist_ok=True)

def main():
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    ensure_dirs(cfg)
    data_dir = Path(cfg["io"]["data_dir"])
    run_name = cfg["io"]["run_name"]
    npz_path = data_dir / f"{run_name}_m1_raw.npz"
    assert npz_path.exists(), f"Missing {npz_path}. Run M1 first."

    loaded = np.load(npz_path, allow_pickle=True)
    tx_list = list(loaded["tx"])
    rx_list = list(loaded["rx"])

    rows = []
    for lane_idx, (tx_blocks, rx_blocks) in enumerate(zip(tx_list, rx_list)):
        # DSP chain per lane
        dsp = dsp_chain(
    rx_blocks,
    ctle_alpha=0.06,
    ffe_taps=7, ffe_mu=5e-4,
    use_dfe=False
)

        # Metrics per block
        B = tx_blocks.shape[0]
        for b in range(B):
            tx_b = tx_blocks[b]              # ground truth symbols (pre-channel)
            y_out = dsp["y_out"][b]          # equalized waveform
            hard_b = dsp["hard_out"][b]      # hard decisions
            # EVM and SNR use hard decisions as reference (decision-directed proxy)
            evm_db = compute_evm_db(y_out, hard_b)
            snr_db = estimate_snr_db(y_out, hard_b)
            eye = eye_metrics_approx(dsp["y_out"][b])

            
            rows.append(dict(
                lane=lane_idx,
                block=b,
                evm_db=evm_db,
                snr_db=snr_db,
                eye_height=eye["eye_height"],
                eye_width=eye["eye_width"],
                taps_var=float(dsp["taps_var"][b]),
            ))

    df = pd.DataFrame(rows)
    out_path = data_dir / f"{run_name}_m2_dsp.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[M2] Wrote DSP metrics: {out_path}")
    print(df.groupby("lane")[["evm_db","snr_db","eye_height"]].mean().round(3))

if __name__ == "__main__":
    main()
