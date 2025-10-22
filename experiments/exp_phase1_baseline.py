from __future__ import annotations
import os,json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from scripts.generate_waveform import PRBSConfig, set_global_seed, generate_multilane
from scripts.channel_model import apply_full_channel_per_lane,apply_crosstalk

def ensure_dirs(cfg):
    Path(cfg["io"]["data_dir"]).mkdir(parents=True,exist_ok=True)
    Path(cfg["io"]["results_dir"]).mkdir(parents=True,exist_ok=True)

def main():
    with open("configs/default.yaml","r") as f:
        cfg=yaml.safe_load(f)
    ensure_dirs(cfg)
    np.set_printoptions(precision=3,suppress=True)
    seed=int(cfg["random_seed"])
    set_global_seed(seed)

    prbs_cfg = PRBSConfig(
        symbols_per_block=int(cfg["lanes"]["symbols_per_block"]),
        blocks_per_run=int(cfg["lanes"]["blocks_per_run"]),
        pam4_levels=np.array(cfg["pam4"]["levels"], dtype=np.float32),
        seed=seed,
    )

    n_lanes = int(cfg["lanes"]["count"])
    tx_blocks_per_lane = generate_multilane(prbs_cfg, n_lanes)

    isi_taps = cfg["channel"]["isi_taps"]
    snr_list = cfg["channel"]["snr_db_per_lane"]
    burst_cfg = cfg["impairments"]["burst_noise"]
    periodic_cfg = cfg["impairments"]["periodic_interference"]

    rx_blocks_per_lane = []
    for lane_idx in range(n_lanes):
        rx_lane = apply_full_channel_per_lane(
            tx_blocks_per_lane[lane_idx],
            isi_taps=isi_taps,
            snr_db=float(snr_list[lane_idx]),
            burst_cfg=burst_cfg,
            periodic_cfg=periodic_cfg,
        )
        rx_blocks_per_lane.append(rx_lane)

    if cfg["crosstalk"]["enabled"]:
        rx_blocks_per_lane = apply_crosstalk(
            rx_blocks_per_lane,
            alpha=float(cfg["crosstalk"]["alpha"]),
            delay_symbols=int(cfg["crosstalk"]["delay_symbols"]),
        )

    data_dir = Path(cfg["io"]["data_dir"])
    run_name = cfg["io"]["run_name"]
    out_path = data_dir / f"{run_name}_m1_raw.npz"
    np.savez_compressed(
        out_path,
        tx=[x for x in tx_blocks_per_lane],
        rx=[x for x in rx_blocks_per_lane],
        cfg_json=json.dumps(cfg),
    )
    print(f"Saved data to: {out_path}")
    rows = []
    for lane_idx in range(n_lanes):
        tx = tx_blocks_per_lane[lane_idx]
        rx = rx_blocks_per_lane[lane_idx]
        rows.append(
            dict(
                lane=lane_idx,
                tx_mean=float(np.mean(tx)),
                tx_std=float(np.std(tx)),
                rx_mean=float(np.mean(rx)),
                rx_std=float(np.std(rx)),
                snr_db=float(snr_list[lane_idx]),
                blocks=int(tx.shape[0]),
                symbols_per_block=int(tx.shape[1]),
            )
        )
    df = pd.DataFrame(rows)
    csv_path = data_dir / f"{run_name}_m1_summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"[M1] Saved raw TX/RX arrays to: {out_path}")
    print(f"[M1] Summary saved to: {csv_path}")
    print(df)

if __name__ == "__main__":
    main()