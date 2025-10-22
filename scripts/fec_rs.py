# scripts/fec_rs.py
from __future__ import annotations
import numpy as np
from typing import Tuple, List
from math import gcd
from reedsolo import RSCodec, ReedSolomonError

# ---------- PAM4 Gray mapping ----------
# Canonical PAM4 Gray levels (index 0..3): 00→-3, 01→-1, 10→+1, 11→+3
_GRAY_LEVELS = np.array([-3., -1., 1., 3.], dtype=np.float32)

_IDX_TO_BITS = np.array([
    [0, 0],  # -3
    [0, 1],  # -1
    [1, 0],  # +1
    [1, 1],  # +3
], dtype=np.uint8)

def bits_to_pam4_levels(bits: np.ndarray) -> np.ndarray:
    """Map bitstream (MSB-first) to PAM4 Gray levels, 2 bits per symbol."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if b.size % 2:
        b = np.pad(b, (0,1))
    pairs = b.reshape(-1, 2)
    idx = (pairs[:, 0] << 1) ^ pairs[:, 1]
    return _GRAY_LEVELS[idx].astype(np.float32)

def pam4_levels_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Nearest-neighbor to canonical Gray levels, then map indices back to bits."""
    s = np.asarray(symbols, dtype=np.float32).reshape(-1)
    idx = np.argmin(np.abs(s[:, None] - _GRAY_LEVELS[None, :]), axis=1)
    b2 = _IDX_TO_BITS[idx]
    return b2.reshape(-1).astype(np.uint8)

# ---------- Bit/byte packing ----------
def bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-b.size) % 8
    if pad:
        b = np.pad(b, (0, pad), constant_values=0)
    B = b.reshape(-1, 8)
    vals = (B[:,0]<<7)|(B[:,1]<<6)|(B[:,2]<<5)|(B[:,3]<<4)|(B[:,4]<<3)|(B[:,5]<<2)|(B[:,6]<<1)|B[:,7]
    return vals.astype(np.uint8)

def bytes_to_bits(by: np.ndarray) -> np.ndarray:
    by = np.asarray(by, dtype=np.uint8).reshape(-1)
    return np.unpackbits(by, bitorder='big').astype(np.uint8)

# ---------- RS(255,k) over GF(2^8) ----------
class RSParams:
    """RS(255,k) with nsym parity bytes. t = nsym/2 byte-corrections."""
    def __init__(self, nsym: int = 32):
        self.nsym = int(nsym)
        self.codec = RSCodec(nsym=self.nsym)  # k = 255 - nsym

def rs_encode_single(params: RSParams, msg_bytes: np.ndarray) -> np.ndarray:
    """Encode exactly one codeword: msg_bytes length must be k."""
    k = 255 - params.nsym
    m = np.asarray(msg_bytes, dtype=np.uint8).reshape(-1)
    if m.size != k:
        raise ValueError(f"msg_bytes must be length k={k}, got {m.size}")
    cw = params.codec.encode(bytes(m.tolist()))  # returns n=255 bytes
    return np.frombuffer(cw, dtype=np.uint8)

def rs_decode_single(params: RSParams, rx_cw: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Decode one received codeword (length 255).
    Returns (decoded_msg_bytes[k], num_corrections_est, fail_flag).
    """
    r = np.asarray(rx_cw, dtype=np.uint8).reshape(-1)
    if r.size != 255:
        raise ValueError(f"rx_cw must be 255 bytes, got {r.size}")
    try:
        dec, _, _ = params.codec.decode(bytes(r.tolist()))  # returns k bytes
        dec_arr = np.frombuffer(dec, dtype=np.uint8)
        # Estimate corrections by difference to re-encoded CW
        re_cw = np.frombuffer(params.codec.encode(dec), dtype=np.uint8)
        corr = int(np.sum(re_cw != r))
        return dec_arr, corr, 0
    except ReedSolomonError:
        return np.zeros(255 - params.nsym, dtype=np.uint8), 255, 1

# ---------- Optional single-CW byte interleaver ----------
def _mod_inverse(a: int, n: int) -> int:
    if gcd(a, n) != 1:
        raise ValueError(f"stride {a} not coprime with {n}")
    return pow(a, -1, n)

def cw_interleave_bytes(cw_bytes: np.ndarray, stride: int = 31) -> np.ndarray:
    n = 255
    b = np.asarray(cw_bytes, dtype=np.uint8).reshape(-1)
    assert b.size == n
    idx = (np.arange(n) * stride) % n
    return b[idx]

def cw_deinterleave_bytes(cw_bytes: np.ndarray, stride: int = 31) -> np.ndarray:
    n = 255
    b = np.asarray(cw_bytes, dtype=np.uint8).reshape(-1)
    assert b.size == n
    inv = _mod_inverse(stride, n)
    idx = (np.arange(n) * inv) % n
    return b[idx]
