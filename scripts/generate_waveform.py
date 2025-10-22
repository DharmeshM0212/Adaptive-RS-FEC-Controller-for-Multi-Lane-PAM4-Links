from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PRBSConfig:
    symbols_per_block: int
    blocks_per_run: int
    pam4_levels: np.ndarray
    seed: int

def set_global_seed(seed: int) -> None:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

def make_prbs_bits(n_bits: int) -> np.ndarray:
    """
    Generate pseudo-random bits (0/1). For now, uniform i.i.d.
    (You can replace with PRBS-31 later if needed.)
    """
    return np.random.randint(0, 2, size=n_bits, dtype=np.uint8)

def map_bits_to_pam4(bits: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Map 2 bits -> one PAM4 symbol using Gray coding:
      00 -> -3
      01 -> -1
      11 -> +1
      10 -> +3
    """
    if bits.size % 2 != 0:
        bits = np.pad(bits, (0, 1), constant_values=0)
    b0 = bits[0::2]
    b1 = bits[1::2]
    idx = (b0 << 1) ^ b1
    pam = levels[idx]
    return pam.astype(np.float32)

def frame_symbols(symbols: np.ndarray, symbols_per_block: int, blocks_per_run: int) -> np.ndarray:
    """
    Cut a long symbol stream into [blocks_per_run, symbols_per_block].
    Pads the tail if needed.
    """
    total_needed = symbols_per_block * blocks_per_run
    if symbols.size < total_needed:
        pad = total_needed - symbols.size
        symbols = np.pad(symbols, (0, pad), mode='wrap')
    return symbols[:total_needed].reshape(blocks_per_run, symbols_per_block)

def generate_lane_symbols(cfg: PRBSConfig) -> np.ndarray:
    """
    Generate a full run worth of PAM4 symbols for a single lane.
    """
    n_syms = cfg.symbols_per_block * cfg.blocks_per_run
    bits = make_prbs_bits(n_syms * 2)  
    pam_syms = map_bits_to_pam4(bits, cfg.pam4_levels)
    blocks = frame_symbols(pam_syms, cfg.symbols_per_block, cfg.blocks_per_run)
    return blocks  

def generate_multilane(cfg: PRBSConfig, n_lanes: int) -> List[np.ndarray]:
    """
    Generate symbol blocks for all lanes.
    Returns list of arrays, each [blocks, symbols_per_block].
    """
    lanes = []
    for lane in range(n_lanes):
        
        _seed = cfg.seed + lane
        np.random.seed(_seed)
        lanes.append(generate_lane_symbols(cfg))
    return lanes
