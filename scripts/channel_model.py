from __future__ import annotations
import numpy as np
from typing import List,Tuple

def _normalize_taps(h:np.ndarray):
    norm=np.sqrt(np.sum(h**2))
    return h/( norm if norm>0 else 1.0)

def apply_isi(x,h_fir):
    h=_normalize_taps(np.asarray(h_fir,dtype=np.float32))
    y=np.zeros_like(x,dtype=np.float32)
    for b in range(x.shape[0]):
        y[b]=np.convolve(x[b],h,mode='same')
    return y

def add_awgn(x,snr_db):
    y=x.astype(np.float32)
    sig_pow=np.mean(y**2)
    snr_lin=10**(snr_db/10)
    noise_pow=sig_pow/max(snr_lin,1e-9)
    noise=np.random.normal(0.0,np.sqrt(noise_pow),size=y.shape).astype(np.float32)
    return y+noise

def add_burst_noise(x,p_burst,length,amplitude):
    y = x.copy().astype(np.float32)
    std = np.std(y)
    if std == 0:
        return y
    for b in range(y.shape[0]):
        i = 0
        while i < y.shape[1]:
            if np.random.rand() < p_burst:
                L = max(1, int(np.random.exponential(length)))
                end = min(y.shape[1], i + L)
                y[b, i:end] += np.random.normal(0, amplitude * std, size=end - i).astype(np.float32)
                i = end
            else:
                i += 1
    return y

def add_periodic_interference(x,f_norm,amplitude):
    y=x.copy().astype(np.float32)
    n=y.shape[1]
    t=np.arrange(n,dtype=np.float32)
    tone=amplitude*np.sin(2*np.pi*f_norm*t)
    for b in range(y.shape[0]):
        y[b]+=tone
    return y

def apply_crosstalk(lanes,alpha,delay_symbols):
    L=len(lanes)
    out=[lanes.copy().astype(np.float32) for lane in lanes]
    for i in range(L):
        j=(i+1)%L
        donor=lanes[j]
        for b in range(donor.shape[0]):
            contributed=np.roll(donor[b],delay_symbols)
            out[i][b]+=alpha*contributed
    return out

def apply_full_channel_per_lane(blocks,isi_taps,snr_db,burst_cfg,periodic_cfg):
    y = apply_isi(blocks, isi_taps)
    y = add_awgn(y, snr_db)
    if burst_cfg.get("enabled", False):
        y = add_burst_noise(
            y,
            p_burst=float(burst_cfg["p_burst"]),
            length=int(burst_cfg["length"]),
            amplitude=float(burst_cfg["amplitude"]),
        )
    if periodic_cfg.get("enabled", False):
        y = add_periodic_interference(
            y,
            f_norm=float(periodic_cfg["f_norm"]),
            amplitude=float(periodic_cfg["amplitude"]),
        )
    return y
