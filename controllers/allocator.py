# controllers/allocator.py
from __future__ import annotations
import numpy as np
from typing import List

__all__ = ["allocate_nsym_from_sqi"]

def allocate_nsym_from_sqi(
    sqi_per_lane: np.ndarray,
    parity_levels: List[int],
    energy_cap: int,
    default_nsym: int,
    target_sqi: float = 0.70,
    tau: float = 6.0,
) -> np.ndarray:
    """
    Allocate per-lane RS parity bytes (nsym) under a global cap, guided by predicted SQI.
    Lower SQI → more parity. Uses softmax over "need" and then snaps to nearest levels,
    while enforcing sum(nsym) ≤ cap (greedy waterfilling).
    """
    sqi = np.asarray(sqi_per_lane, dtype=float)
    L = sqi.size
    levels = np.array(sorted(parity_levels), dtype=int)

    # If levels are empty or cap invalid, fall back to defaults safely
    if levels.size == 0:
        return np.full(L, default_nsym, dtype=int)
    if energy_cap <= 0:
        return np.full(L, levels[0], dtype=int)

    # Need score: higher when sqi << target
    need = np.clip((target_sqi - sqi), 0.0, 1.0)

    # If everyone healthy, return defaults but clip to cap
    if np.allclose(need, 0.0):
        out = np.full(L, default_nsym, dtype=int)
        total = int(out.sum())
        if total > energy_cap:
            # reduce healthiest first until under cap
            order = np.argsort(-sqi)  # healthiest first
            i = 0
            while total > energy_cap and i < 10 * L:
                lane = order[i % L]
                cur = out[lane]
                idx = np.where(levels == cur)[0]
                if idx.size and idx[0] > 0:
                    out[lane] = int(levels[idx[0] - 1])
                    total = int(out.sum())
                i += 1
        return out

    # Softmax focus on need
    logits = need * tau
    w = np.exp(logits - np.max(logits))
    w = w / np.sum(w)

    # Start from minimum level for all lanes
    out = np.full(L, levels[0], dtype=int)
    budget = energy_cap - int(out.sum())
    if budget <= 0:
        return out

    # Track current level index per lane
    cur_idx = np.zeros(L, dtype=int)

    # Greedy waterfilling: allocate next level to lane with largest weight per cost
    while budget > 0:
        gains = np.zeros(L, dtype=int)
        for i in range(L):
            if cur_idx[i] < len(levels) - 1:
                gains[i] = int(levels[cur_idx[i] + 1] - levels[cur_idx[i]])
            else:
                gains[i] = 0

        # No further upgrades possible
        if np.all(gains == 0):
            break

        mask = gains > 0
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            break

        lane_scores = w[idxs] / np.maximum(gains[idxs], 1)
        lane_pick = idxs[int(np.argmax(lane_scores))]
        cost = gains[lane_pick]

        if cost <= budget:
            cur_idx[lane_pick] += 1
            out[lane_pick] = int(levels[cur_idx[lane_pick]])
            budget -= cost
        else:
            break

    return out
