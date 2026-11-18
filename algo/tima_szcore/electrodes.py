"""Standard 10-20 electrode 2D scalp coordinates + bipolar-pair geometry.

The montage-agnostic encoder represents each channel by WHERE it sits on the
scalp (not by a fixed index). For a bipolar pair (anode, cathode) we use:
  - the midpoint (x, y)            -> "where" the channel is
  - the direction vector (dx, dy)  -> "orientation" of the derivation
so each channel gets a 4-D geometric descriptor that is montage-independent and
shared across datasets. New montages just map to new coordinates.
"""
from __future__ import annotations
import numpy as np

# Approx. 2D projected 10-20 positions (nose = +y, right ear = +x), unit-ish disk.
ELECTRODES_2D = {
    "FP1": (-0.31, 0.95), "FP2": (0.31, 0.95),
    "F7": (-0.81, 0.59),  "F3": (-0.40, 0.66), "FZ": (0.0, 0.71), "F4": (0.40, 0.66), "F8": (0.81, 0.59),
    "T7": (-1.00, 0.0),   "T3": (-1.00, 0.0),  "C3": (-0.50, 0.0), "CZ": (0.0, 0.0), "C4": (0.50, 0.0),
    "T8": (1.00, 0.0),    "T4": (1.00, 0.0),
    "P7": (-0.81, -0.59), "T5": (-0.81, -0.59), "P3": (-0.40, -0.66), "PZ": (0.0, -0.71),
    "P4": (0.40, -0.66),  "P8": (0.81, -0.59),  "T6": (0.81, -0.59),
    "O1": (-0.31, -0.95), "O2": (0.31, -0.95),
}

# The 18 CHB-MIT bipolar pairs (must match mvafa_szcore.eeg_io.BIPOLAR_PAIRS order).
BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FZ", "CZ"), ("CZ", "PZ"),
]


# Canonical 19-channel referential 10-20 montage (used natively by Siena & TUSZ).
REFERENTIAL_1020 = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7", "C3", "CZ", "C4", "T8",
    "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
]


def bipolar_geometry(pairs=BIPOLAR_PAIRS) -> np.ndarray:
    """Return (C, 4) geometric descriptor [mid_x, mid_y, dir_x, dir_y] per pair."""
    g = []
    for a, b in pairs:
        ax, ay = ELECTRODES_2D[a]
        bx, by = ELECTRODES_2D[b]
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        dx, dy = (bx - ax), (by - ay)
        n = (dx * dx + dy * dy) ** 0.5 + 1e-8
        g.append([mx, my, dx / n, dy / n])
    return np.asarray(g, dtype=np.float32)


def referential_geometry(names) -> np.ndarray:
    """Return (C, 4) descriptor [x, y, 0, 0] for referential (monopolar) channels.

    A referential channel is a single electrode: a POINT on the scalp with no
    derivation direction, so the last two dims are zero. Critically this lives in
    the SAME (x, y) coordinate frame as `bipolar_geometry`'s midpoints, so a model
    keyed on coordinates can relate an 18-bipolar montage to a 19-referential one
    (a channel-INDEX encoding cannot). Unknown names map to the origin.
    """
    g = []
    for nm in names:
        x, y = ELECTRODES_2D.get(nm.upper(), (0.0, 0.0))
        g.append([x, y, 0.0, 0.0])
    return np.asarray(g, dtype=np.float32)
