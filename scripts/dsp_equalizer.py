# scripts/dsp_equalizer.py  (stable FFE, DFE disabled by default)
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

def ctle_iir(x: np.ndarray, alpha_hp: float = 0.06) -> np.ndarray:
    a = float(alpha_hp)
    y = np.zeros_like(x, dtype=np.float32)
    for b in range(x.shape[0]):
        prev_y = 0.0
        prev_x = 0.0
        for n in range(x.shape[1]):
            cur = x[b, n]
            out = cur - (1.0 - a) * prev_x + a * prev_y
            y[b, n] = out
            prev_y = out
            prev_x = cur
        # rescale to input RMS
        rms_in  = np.sqrt(np.mean(x[b]**2) + 1e-12)
        rms_out = np.sqrt(np.mean(y[b]**2) + 1e-12)
        if rms_out > 0:
            y[b] *= (rms_in / rms_out)
    return y

def _block_norm(xb: np.ndarray) -> Tuple[np.ndarray, float]:
    rms = np.sqrt(np.mean(xb**2) + 1e-12)
    return (xb / max(rms, 1e-9)).astype(np.float32), float(max(rms, 1e-9))

def ffe_lms_stable(x: np.ndarray, d: np.ndarray, n_taps: int = 7,
                   mu: float = 5e-4, leak: float = 5e-6, wclip: float = 1.5
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conservative LMS-FFE:
      - per-block input normalization
      - unit center-tap init
      - small step + leakage
      - weight-norm clipping to avoid divergence
    """
    B, N = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    taps = np.zeros((B, n_taps), dtype=np.float32)
    c = n_taps // 2

    for b in range(B):
        xb, scale = _block_norm(x[b])
        db, _     = _block_norm(d[b])
        w = np.zeros(n_taps, dtype=np.float32); w[c] = 1.0
        buf = np.zeros(n_taps, dtype=np.float32)

        for n in range(N):
            buf[1:] = buf[:-1]
            buf[0]  = xb[n]
            y_hat   = float(np.dot(w, buf))
            e       = db[n] - y_hat
            # LMS update with leakage
            w = (1.0 - leak) * w + mu * e * buf
            # clip weight norm
            wn = np.linalg.norm(w) + 1e-12
            if wn > wclip:
                w *= (wclip / wn)
            y[b, n] = y_hat * scale  # restore original scale

        taps[b] = w
    return y, taps

def hard_decision_pam4(x: np.ndarray) -> np.ndarray:
    levels = np.array([-3., -1.,  1.,  3.], dtype=np.float32)
    idx = np.argmin(np.abs(x[..., None] - levels[None, None, :]), axis=-1)
    return levels[idx].astype(np.float32)

def crude_cdr_and_downsample(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    timing_err = np.zeros_like(x, dtype=np.float32)
    return x.astype(np.float32), timing_err

def dsp_chain(blocks_rx: np.ndarray,
              ctle_alpha: float = 0.06,
              ffe_taps: int = 7, ffe_mu: float = 5e-4,
              use_dfe: bool = False, dfe_taps: int = 0, dfe_mu: float = 0.0
              ) -> Dict[str, np.ndarray]:
    """
    CTLE -> CDR -> stable FFE (no DFE for M2).
    """
    y_ctle, timing_err = crude_cdr_and_downsample(ctle_iir(blocks_rx, alpha_hp=ctle_alpha))
    ref0  = hard_decision_pam4(y_ctle)
    y_ffe, w_ffe = ffe_lms_stable(y_ctle, ref0, n_taps=ffe_taps, mu=ffe_mu)

    y_out   = y_ffe
    hard_out= hard_decision_pam4(y_out)
    taps_var= np.var(w_ffe, axis=1)

    return dict(
        y_ctle=y_ctle, y_sym=y_ctle, timing_err=timing_err,
        y_ffe=y_ffe, w_ffe=w_ffe,
        y_out=y_out, w_dfe=np.zeros((blocks_rx.shape[0],0),dtype=np.float32),
        hard_out=hard_out,
        taps_var=taps_var.astype(np.float32),
    )
