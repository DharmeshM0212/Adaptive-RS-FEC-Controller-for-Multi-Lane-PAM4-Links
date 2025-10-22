# experiments/exp_phase3_rs_pipeline.py
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from scripts.generate_waveform import set_global_seed, make_prbs_bits
from scripts.fec_rs import (
    RSParams, rs_encode_single, rs_decode_single,
    bits_to_pam4_levels, pam4_levels_to_bits,
    bits_to_bytes, bytes_to_bits,
    cw_interleave_bytes, cw_deinterleave_bytes,
)

# ---------- simple deterministic channel ----------
def apply_channel_symbol_spaced(x: np.ndarray,
                                isi_taps: list[float],
                                snr_db: float,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Symbol-spaced causal FIR + AWGN. Keeps same length as x via zero-padding.
    SNR is defined on the *post-ISI clean* waveform.
    """
    x = x.astype(np.float32).reshape(-1)
    h = np.asarray(isi_taps, dtype=np.float32).reshape(-1)
    L = h.size
    # zero-padded convolution (same-length)
    y = np.zeros_like(x, dtype=np.float32)
    buf = np.zeros(L, dtype=np.float32)
    for n in range(x.size):
        buf[1:] = buf[:-1]
        buf[0]  = x[n]
        y[n]    = float(np.dot(h, buf))
    # AWGN to reach target SNR on clean y
    p_sig = float(np.mean(y**2) + 1e-12)
    snr_lin = 10.0**(snr_db/10.0)
    sigma2 = p_sig / snr_lin
    noise = rng.normal(0.0, np.sqrt(sigma2), size=y.shape).astype(np.float32)
    return y + noise

# ---------- deterministic MMSE-FFE trained on pilot ----------
def design_mmse_ffe_from_pilot(rx_pilot: np.ndarray,
                               tx_pilot: np.ndarray,
                               n_taps: int = 9,
                               reg_lambda: float = 1e-3) -> np.ndarray:
    """
    Solve w = argmin ||X w - d||^2 + λ||w||^2
    where X is Toeplitz of rx_pilot, d = tx_pilot (canonical {-3,-1,1,3}).
    """
    x = rx_pilot.astype(np.float32).reshape(-1)
    d = tx_pilot.astype(np.float32).reshape(-1)
    N = x.size
    L = n_taps
    # Build X (N x L) with causal taps [x[n], x[n-1], ...]
    X = np.zeros((N, L), dtype=np.float32)
    for n in range(N):
        for k in range(L):
            idx = n - k
            X[n, k] = x[idx] if idx >= 0 else 0.0
    # Ridge regression closed-form
    XtX = X.T @ X
    Xtd = X.T @ d
    w = np.linalg.solve(XtX + reg_lambda * np.eye(L, dtype=np.float32), Xtd)
    return w.astype(np.float32)

def apply_ffe(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Causal FIR apply (same convention as training)."""
    x = x.astype(np.float32).reshape(-1)
    w = w.astype(np.float32).reshape(-1)
    L = w.size
    y = np.zeros_like(x, dtype=np.float32)
    buf = np.zeros(L, dtype=np.float32)
    for n in range(x.size):
        buf[1:] = buf[:-1]
        buf[0]  = x[n]
        y[n]    = float(np.dot(w, buf))
    return y

# ---------- robust AGC/offset from pilot ----------
def affine_fit(y: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """
    Fit y' = a*y + b to match target d in least-squares sense.
    Returns (a, b).
    """
    y = y.astype(np.float32).reshape(-1)
    d = d.astype(np.float32).reshape(-1)
    A = np.stack([y, np.ones_like(y)], axis=1)
    # Solve min ||A [a b]^T - d||^2
    sol, *_ = np.linalg.lstsq(A, d, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    return a, b

# ---------- deterministic pilot ----------
def make_pilot(length: int) -> np.ndarray:
    # Repeat canonical pattern so every level is well represented
    base = np.array([-3., -1., 1., 3.], dtype=np.float32)
    reps = int(np.ceil(length / base.size))
    p = np.tile(base, reps)[:length]
    return p

# ---------- metrics ----------
def evm_db_against_decisions(y: np.ndarray, hard_levels: np.ndarray) -> float:
    e = y.astype(np.float32) - hard_levels.astype(np.float32)
    s = hard_levels.astype(np.float32)
    num = float(np.mean(e*e) + 1e-12)
    den = float(np.mean(s*s) + 1e-12)
    return -10.0 * np.log10(num / den)

def snr_db_against_decisions(y: np.ndarray, hard_levels: np.ndarray) -> float:
    s = hard_levels.astype(np.float32)
    e = y.astype(np.float32) - s
    p_s = float(np.mean(s*s) + 1e-12)
    p_e = float(np.mean(e*e) + 1e-12)
    return 10.0 * np.log10(p_s / p_e)

# ---------- main ----------
def main():
    # 1) Load config
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(cfg["io"]["data_dir"]); data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["io"]["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["io"]["run_name"]

    seed = int(cfg["random_seed"]); rng0 = np.random.default_rng(seed)

    n_lanes = int(cfg["lanes"]["count"])
    PILOT = int(cfg["lanes"]["pilot_len"])
    DATA  = int(cfg["lanes"]["data_len"])
    assert DATA == 1020, "data_len must be exactly 1020 (one RS(255,·) codeword)"
    BPR  = int(cfg["lanes"]["blocks_per_run"])

    isi_taps = cfg["channel"]["isi_taps"]
    snr_list = cfg["channel"]["snr_db_per_lane"]
    print("[DEBUG] cfg.channel.isi_taps =", isi_taps)
    print("[DEBUG] cfg.channel.snr_db_per_lane =", snr_list)
    print("[DEBUG] burst.enabled =", cfg["impairments"]["burst_noise"]["enabled"],
          "tone.enabled =", cfg["impairments"]["periodic_interference"]["enabled"])

    rs_params = RSParams(nsym=int(cfg["fec"]["rs_nsym"]))
    use_ilv  = bool(cfg["fec"]["interleave_bytes"])
    stride   = int(cfg["fec"]["interleave_stride"])

    # 2) Fixed pilot sequence
    pilot_levels = make_pilot(PILOT)  # shape (PILOT,)

    rows = []

    for lane in range(n_lanes):
        rng_lane = np.random.default_rng(seed + lane)

        for b in range(BPR):
            # ----- TX: one RS CW → 1020 PAM4 data symbols -----
            k_bytes = 255 - rs_params.nsym
            pay_bits = make_prbs_bits(k_bytes * 8)               # exact payload size
            pay_bytes = bits_to_bytes(pay_bits)
            cw_bytes  = rs_encode_single(rs_params, pay_bytes)   # 255 bytes (data+parity)

            if use_ilv:
                cw_bytes_tx = cw_interleave_bytes(cw_bytes, stride=stride)
            else:
                cw_bytes_tx = cw_bytes

            cw_bits_tx = bytes_to_bits(cw_bytes_tx)              # 2040 bits
            data_levels = bits_to_pam4_levels(cw_bits_tx)        # 1020 symbols in {-3,-1,1,3}

            tx_block = np.concatenate([pilot_levels, data_levels])  # [PILOT+DATA]

            # ----- Channel (deterministic) -----
            y = apply_channel_symbol_spaced(
                tx_block, isi_taps=isi_taps,
                snr_db=float(snr_list[lane]),
                rng=rng_lane
            )

            # ----- Equalizer training on PILOT only (MMSE FFE) -----
            y_pilot = y[:PILOT]
            d_pilot = pilot_levels

            w_ffe = design_mmse_ffe_from_pilot(y_pilot, d_pilot, n_taps=9, reg_lambda=1e-3)
            y_eq  = apply_ffe(y, w_ffe)   # equalize entire block

            # ----- AGC/offset from pilot only -----
            y_eq_pilot = y_eq[:PILOT]
            a, b = affine_fit(y_eq_pilot, d_pilot)  # fit to canonical levels
            y_cal = a * y_eq + b

            # ----- Hard slicing on DATA portion -----
            y_data = y_cal[PILOT : PILOT + DATA]
            # Decision regions around canonical levels
            # thresholds at [-2, 0, +2] in the *calibrated* domain
            hard = np.empty_like(y_data)
            hard[y_data < -2.0] = -3.0
            hard[(y_data >= -2.0) & (y_data < 0.0)]  = -1.0
            hard[(y_data >= 0.0)  & (y_data < 2.0)]  =  1.0
            hard[y_data >= 2.0] =  3.0

            # ----- Metrics on DATA portion -----
            ser = float(np.mean(hard != data_levels))
            evm_db = evm_db_against_decisions(y_data, hard)
            snr_db = snr_db_against_decisions(y_data, hard)

            # ----- RS decode (DATA → bits → bytes → (de)interleave → decode) -----
            rx_bits  = pam4_levels_to_bits(hard)     # 2040 bits
            rx_bytes = bits_to_bytes(rx_bits)        # 255 bytes (codeword)

            if use_ilv:
                rx_cw = cw_deinterleave_bytes(rx_bytes, stride=stride)
            else:
                rx_cw = rx_bytes

            dec_bytes, rs_corr, rs_fail = rs_decode_single(rs_params, rx_cw)

            rows.append(dict(
                lane=lane, block=b,
                ser=ser, evm_db=evm_db, snr_db=snr_db,
                rs_corr_mean=float(rs_corr), rs_corr_p95=float(rs_corr),
                rs_fail_cw=int(rs_fail),
            ))

    df = pd.DataFrame(rows)
    out_path = Path(cfg["io"]["data_dir"]) / f"{run_name}_m3_metrics.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[M3] Wrote deterministic DSP+FEC metrics: {out_path}")
    print(
        df.groupby("lane")[['ser','evm_db','snr_db','rs_corr_mean','rs_fail_cw']]
          .mean().round(4)
    )

if __name__ == "__main__":
    main()
