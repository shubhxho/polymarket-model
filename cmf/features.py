from __future__ import annotations

import numpy as np

HISTORY = 64
FAST_DIM = 24
SLOW_DIM = 16
POS_DIM = 8
LAG_DIM = 8
RAW_FAST_DIM = 16
RAW_SLOW_DIM = 12

_HAS_NATIVE = False
_native = None

try:
    from cmf import _cmf_native as _native  # type: ignore

    HISTORY = int(_native.HISTORY)
    FAST_DIM = int(_native.FAST_DIM)
    SLOW_DIM = int(_native.SLOW_DIM)
    POS_DIM = int(_native.POS_DIM)
    LAG_DIM = int(_native.LAG_DIM)
    RAW_FAST_DIM = int(_native.RAW_FAST_DIM)
    RAW_SLOW_DIM = int(_native.RAW_SLOW_DIM)
    _HAS_NATIVE = True
except ImportError:
    _native = None


def has_native() -> bool:
    return _HAS_NATIVE


def featurize_episode(fast_raw: np.ndarray, slow_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw ticks -> (fast [T,24], slow [T,16], lag [T,8]). Prefers the C++ engine."""
    fast_raw = np.ascontiguousarray(fast_raw, dtype=np.float32)
    slow_raw = np.ascontiguousarray(slow_raw, dtype=np.float32)
    if _HAS_NATIVE:
        fast, slow, lag = _native.featurize_episode(fast_raw, slow_raw)
        return np.asarray(fast), np.asarray(slow), np.asarray(lag)
    return _featurize_python(fast_raw, slow_raw)


def window_at(seq: np.ndarray, t: int, history: int) -> np.ndarray:
    """Causal history ending at t, zero-padded on the left."""
    start = t - history + 1
    if start >= 0:
        return seq[start : t + 1]
    pad = np.zeros((-start, seq.shape[1]), dtype=seq.dtype)
    return np.concatenate([pad, seq[: t + 1]], axis=0)


def lacuna_vector(fast: np.ndarray, slow: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Map rich features onto LACUNA's 18-dim observation for the baseline."""
    return np.array(
        [
            fast[0],
            fast[1],
            fast[3],
            fast[8],
            fast[9],
            fast[10],
            fast[11],
            slow[2],
            fast[16],
            fast[15],
            fast[6],
            fast[7],
            pos[0],
            pos[1],
            pos[2],
            pos[5],
            np.float32(fast[6] > 0.35),
            np.float32(abs(fast[3]) > 0.25),
        ],
        dtype=np.float32,
    )


def _safe_div(a: float, b: float) -> float:
    return float(a / (abs(b) + 1e-8))


def _featurize_python(fast_raw: np.ndarray, slow_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference path used only if the extension is not built."""
    t = fast_raw.shape[0]
    fast = np.zeros((t, FAST_DIM), dtype=np.float32)
    slow = np.zeros((t, SLOW_DIM), dtype=np.float32)
    lag = np.zeros((t, LAG_DIM), dtype=np.float32)
    fm = fast_raw[:, 1]
    sm = slow_raw[:, 1]
    fr = np.zeros(t, dtype=np.float32)
    sr = np.zeros(t, dtype=np.float32)
    fr[1:] = np.diff(fm) / (np.abs(fm[:-1]) + 1e-8)
    sr[1:] = np.diff(sm) / (np.abs(sm[:-1]) + 1e-8)
    cvd = np.cumsum(fast_raw[:, 6] * fast_raw[:, 7])
    for i in range(t):
        def wr(x: np.ndarray, n: int) -> float:
            j = max(0, i - n)
            return _safe_div(x[i] - x[j], x[j])

        rv = float(np.std(fr[max(0, i - 20) : i + 1])) if i > 1 else 0.0
        rv5 = float(np.std(fr[max(0, i - 5) : i + 1])) if i > 1 else 0.0
        fast[i, 0] = np.tanh(wr(fm, 1) * 80)
        fast[i, 1] = np.tanh(wr(fm, 5) * 40)
        fast[i, 2] = np.tanh(wr(fm, 15) * 25)
        fast[i, 3] = np.tanh(wr(fm, 30) * 18)
        fast[i, 4] = np.tanh(wr(fm, 60) * 12)
        fast[i, 6] = np.tanh(rv * 80)
        fast[i, 7] = np.tanh(_safe_div(rv5, rv + 1e-8) - 1.0)
        fast[i, 10] = np.tanh(fast_raw[i, 7])
        fast[i, 11] = np.tanh((cvd[i] - (cvd[i - 1] if i else 0.0)) * 1e-3)
        fast[i, 15] = float(fast_raw[i, 15])
        fast[i, 19] = np.tanh(fast_raw[i, 10] * 200)
        fast[i, 22] = np.tanh(fast_raw[i, 14] * 200)
        fast[i, 23] = float(np.clip(fast_raw[i, 15] * np.sign(fr[i]), -1, 1))
        mid = sm[i]
        slow[i, 0] = float(np.clip(2 * mid - 1, -1, 1))
        slow[i, 2] = np.tanh(_safe_div(slow_raw[i, 3] - slow_raw[i, 2], max(mid, 0.05)) * 20)
        slow[i, 7] = np.tanh(wr(sm, 3) * 25)
        slow[i, 13] = float(np.clip(slow_raw[i, 10], 0, 1))
        slow[i, 14] = float(np.clip(2 * (mid - 0.5), -1, 1))
        slow[i, 15] = np.tanh(slow_raw[i, 11] / 8.0)
        if i >= 16:
            for li, k in enumerate((0, 1, 2, 4, 8, 16)):
                a = fr[: i + 1 - k]
                b = sr[k : i + 1]
                n = min(len(a), len(b), 48)
                if n > 6:
                    lag[i, li] = float(np.clip(np.corrcoef(a[-n:], b[-n:])[0, 1], -1, 1))
            lag[i, 6] = float(np.clip(np.nanmax(lag[i, :6]) - lag[i, 0], -1, 1))
            lag[i, 7] = float(np.argmax(lag[i, :6]) / 5.0)
    return fast, slow, lag
