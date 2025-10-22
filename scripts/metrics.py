# scripts/metrics.py  (REPLACE the functions below)
from __future__ import annotations
import numpy as np
from typing import Dict

def compute_evm_db(rx_syms: np.ndarray, ref_syms: np.ndarray) -> float:
    """
    EVM_dB = 10*log10( sum|rx - ref|^2 / sum|ref|^2 ).
    """
    num = np.sum((rx_syms - ref_syms) ** 2, dtype=np.float64)
    den = np.sum((ref_syms) ** 2, dtype=np.float64) + 1e-12
    evm_lin = num / den
    return float(10.0 * np.log10(max(evm_lin, 1e-18)))

def estimate_snr_db(rx_syms: np.ndarray, ref_syms: np.ndarray) -> float:
    """
    SNR ≈ Var(ref)/Var(error) (blockwise). Should be positive when equalization is reasonable.
    """
    err = rx_syms - ref_syms
    s_var = np.var(ref_syms, dtype=np.float64) + 1e-12
    n_var = np.var(err, dtype=np.float64) + 1e-12
    snr = s_var / n_var
    return float(10.0 * np.log10(max(snr, 1e-18)))

def eye_metrics_approx(equalized_waveform: np.ndarray, levels=(-3., -1., 1., 3.)) -> Dict[str, float]:
    """
    Eye metrics from equalized samples:
      - Eye height: min gap between adjacent cluster means minus 2*avg std (>=0).
      - Eye width (proxy): fraction of samples confidently near level centers.
    """
    wf = equalized_waveform.flatten().astype(np.float32)
    lv = np.array(levels, dtype=np.float32)
    idx = np.argmin(np.abs(wf[:, None] - lv[None, :]), axis=1)

    centers = []
    stds = []
    conf_mask = np.zeros_like(wf, dtype=bool)
    for k in range(len(lv)):
        vals = wf[idx == k]
        if vals.size < 20:
            continue
        centers.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        conf_mask[idx == k] = np.abs(vals - np.mean(vals)) < (1.5 * np.std(vals) + 1e-6)

    if len(centers) >= 2:
        centers = np.array(sorted(centers))
        stds = np.array(sorted(stds))  # rough pairing
        min_gap = float(np.min(np.diff(centers)))
        avg_std = float(np.mean(stds)) if stds.size else 0.0
        eye_h = max(min_gap - 2.0 * avg_std, 0.0)
    else:
        eye_h = 0.0

    eye_w = float(np.mean(conf_mask))  # higher = wider "open" region proxy
    return dict(eye_height=eye_h, eye_width=eye_w)
