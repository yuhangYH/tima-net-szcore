"""Per-channel features for the montage-agnostic encoder.

History (the root-cause fix). The original descriptor was 100% scale-invariant
by construction, so it transferred across hardware but capped AUROC near 0.60 on
the hard held-out cohorts. The ceiling did not move with the model, which located
the bottleneck in the features, not the network. The two cues that separate
seizure from background in the literature, absolute line length and absolute
power, had both been removed. Line length was even computed and then dropped from
the vector. See the LODO gate results for the plateau evidence.

The descriptor now keeps the scale-invariant block AND adds an amplitude-aware
block, so we can test whether the invariance was costing discriminability:

  Scale-invariant block (indices 0..8, unchanged and backward compatible)
  - relative band power ratios (delta/theta/alpha/beta/gamma; sum to 1)   [5]
  - spectral entropy of the normalized PSD                               [1]
  - Hjorth mobility & complexity (ratios of variances -> scale-free)     [2]
  - zero-crossing rate                                                   [1]

  Amplitude-aware block (indices 9..11, default 12-D)
  - relative line length (mean|d1| / std), scale-invariant, restored     [1]
  - log absolute line length, log1p(mean|d1|), amplitude/morphology      [1]
  - log absolute band power, log(sum of band power), amplitude/energy    [1]

  Topological block (indices 12..19, opt-in via TIMA_FEAT_TDA=1)          [8]
  - per-channel persistent homology of a Takens delay embedding.

Feature axes and how they are switched (independent env flags):

  TIMA_FEAT_LEGACY9=1  -> 9-D  (old scale-invariant baseline)
  (default)            -> 12-D (amplitude-aware root-cause fix)
  TIMA_FEAT_TDA=1      -> +8   topological dims appended (e.g. 12 -> 20)

The topological block is PER CHANNEL on purpose. Persistent homology of one
channel's own delay embedding does not depend on which other electrodes are
present, so it stays montage-agnostic and can be concatenated into the electrode
token without breaking the cross-montage claim. A cross-channel point cloud would
be montage-dependent and is deliberately avoided here. TDA is a FEATURE-axis
change: all three arms (index / coord / coord+CORAL) consume the same fused
token, so the whole three-arm gate is simply re-run at the new baseline.

The feature dimension is written into the cache tag (see train_tima.py), so a
descriptor change forces recomputation instead of loading a stale cache; 12-D and
20-D never share a cache. TDA is heavy (ripser per channel per window), so rely
on the existing multi-worker precompute and the on-disk cache.

Per channel -> a DIM-D descriptor. Returns (C, DIM) for a (C, T) window.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.signal import welch

SFREQ = 256.0
_BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]   # delta theta alpha beta gamma

# A/B switch for the root-cause experiment. TIMA_FEAT_LEGACY9=1 emits the old
# 100% scale-invariant 9-D vector (indices 0..8), so the harness can run the
# legacy baseline and the 12-D descriptor under identical folds/seeds/caps and
# isolate the feature change from any setup change. Default is the full 12-D.
LEGACY9 = os.environ.get("TIMA_FEAT_LEGACY9", "") == "1"

# Opt-in topological block. Orthogonal to the 9-D/12-D switch: it appends K_TDA
# persistent-homology dims to whichever base descriptor is active.
TDA = os.environ.get("TIMA_FEAT_TDA", "") == "1"
K_TDA = 8

# Scale-free TDA variant (TIMA_TDA_SCALEFREE=1). The default TDA block keeps
# amplitude (persistence scales with signal magnitude), which cost held-out AUROC
# on Siena, whose referential recordings sit at a different amplitude scale than
# the other cohorts. When this flag is set, each channel's series is robustly
# standardized (median / MAD) BEFORE the Takens embedding, so the point cloud and
# every persistence value are amplitude-invariant while the 8-D vectorization is
# unchanged. The variant is a distinct feature set from amplitude-keeping TDA, so
# it must not share a cache: CHAN_FEAT_TAG below suffixes the cache key with 'sf'.
TDA_SCALEFREE = os.environ.get("TIMA_TDA_SCALEFREE", "") == "1"

# Extra cache-key suffix, appended after fNN in train_tima.py, so two descriptors
# that share the same dimension but differ in construction never collide.
CHAN_FEAT_TAG = ("sf" if (TDA and TDA_SCALEFREE) else "")

# Takens delay embedding for the per-channel point cloud.
_TDA_M = 3          # embedding dimension
_TDA_TAU = 8        # delay in samples (fixed for reproducibility)
_TDA_NMAX = 200     # cap the point cloud so ripser (maxdim=1) stays cheap

_BASE_DIM = 9 if LEGACY9 else 12
CHAN_FEAT_DIM = _BASE_DIM + (K_TDA if TDA else 0)


def _persistence_entropy(lifetimes: np.ndarray) -> float:
    s = lifetimes.sum()
    if s <= 1e-12 or lifetimes.size == 0:
        return 0.0
    p = lifetimes / s
    return float(-(p * np.log(p + 1e-12)).sum())


def _tda_features(sig: np.ndarray) -> list:
    """8-D per-channel topological descriptor from a Takens delay embedding.

    Amplitude is kept (no per-window normalization) so persistence scales with
    signal amplitude, consistent with the 12-D amplitude-aware philosophy;
    magnitude terms are log-compressed and entropy/ratio terms are scale-free.
    """
    try:
        from ripser import ripser
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "TIMA_FEAT_TDA=1 requires 'ripser' (pip install ripser)."
        ) from exc

    n = sig.size - (_TDA_M - 1) * _TDA_TAU
    if n < 10:
        return [0.0] * K_TDA
    if TDA_SCALEFREE:
        # robust per-channel standardization -> amplitude-invariant point cloud
        med = np.median(sig)
        mad = np.median(np.abs(sig - med)) * 1.4826 + 1e-8
        sig = (sig - med) / mad
    # Takens embedding -> point cloud in R^m
    cloud = np.stack([sig[i * _TDA_TAU: i * _TDA_TAU + n] for i in range(_TDA_M)], axis=1)
    # subsample by stride to bound ripser cost while keeping temporal coverage
    if cloud.shape[0] > _TDA_NMAX:
        stride = int(np.ceil(cloud.shape[0] / _TDA_NMAX))
        cloud = cloud[::stride]
    try:
        dgms = ripser(cloud, maxdim=1)["dgms"]
    except Exception:                                # pragma: no cover - degenerate window
        return [0.0] * K_TDA

    h0, h1 = dgms[0], dgms[1]
    # H0: finite bars (birth=0, death=merge distance); drop the infinite bar
    h0_death = h0[:, 1]
    h0_death = h0_death[np.isfinite(h0_death)]
    h0_total = float(h0_death.sum())
    h0_ent = _persistence_entropy(h0_death)
    # H1: loops
    if h1.size:
        h1_life = h1[:, 1] - h1[:, 0]
        h1_life = h1_life[np.isfinite(h1_life)]
    else:
        h1_life = np.empty(0)
    if h1_life.size:
        h1_count = float(h1_life.size)
        h1_max = float(h1_life.max())
        h1_total = float(h1_life.sum())
        h1_ent = _persistence_entropy(h1_life)
        h1_mean_birth = float(h1[:, 0][np.isfinite(h1[:, 1])].mean())
    else:
        h1_count = h1_max = h1_total = h1_ent = h1_mean_birth = 0.0

    return [
        h1_count,                                    # 1 number of loops (scale-free)
        float(np.log1p(h1_max)),                     # 2 most persistent cycle (log)
        float(np.log1p(h1_total)),                   # 3 total H1 persistence (log)
        h1_ent,                                      # 4 H1 persistence entropy (scale-free)
        float(np.log1p(max(h1_mean_birth, 0.0))),    # 5 mean H1 birth scale (log)
        float(np.log1p(h0_total)),                   # 6 total H0 persistence (log)
        h0_ent,                                      # 7 H0 persistence entropy (scale-free)
        float(h1_total / (h0_total + 1e-8)),         # 8 H1/H0 ratio (scale-free)
    ]


def _per_channel(sig: np.ndarray) -> list:
    sig = np.asarray(sig, dtype=np.float64)
    std = sig.std() + 1e-8
    # --- band power ---
    nper = min(128, len(sig))
    freqs, pxx = welch(sig, fs=SFREQ, nperseg=nper, noverlap=nper // 2)
    bp = []
    for lo, hi in _BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp.append(float(pxx[m].sum()))
    bp = np.asarray(bp)
    tot = bp.sum() + 1e-12
    ratios = (bp / tot).tolist()                       # 5 scale-invariant ratios (sum=1)
    p = bp / tot
    spec_ent = float(-(p * np.log(p + 1e-12)).sum())   # 1 spectral entropy
    # --- Hjorth (scale-free) ---
    d1 = np.diff(sig); d2 = np.diff(d1)
    v0 = sig.var() + 1e-12; v1 = d1.var() + 1e-12; v2 = d2.var() + 1e-12
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / (mobility + 1e-12)  # 2 (mobility, complexity)
    # --- zero-crossing rate (scale-free) ---
    zcr = float(np.mean(np.abs(np.diff(np.sign(sig))) > 0))   # 1
    # --- line length, both views ---
    mean_abs_d1 = float(np.mean(np.abs(d1)))
    ll_rel = mean_abs_d1 / std                          # 1 scale-invariant (restored)
    ll_abs = float(np.log1p(mean_abs_d1))               # 1 amplitude-aware
    # --- absolute band power (amplitude-aware, log-compressed) ---
    log_pow = float(np.log(tot))                        # 1 amplitude-aware
    if LEGACY9:
        # old 100% scale-invariant descriptor (the plateaued baseline)
        feats = ratios + [spec_ent, float(mobility), float(complexity), zcr]
    else:
        feats = ratios + [spec_ent, float(mobility), float(complexity), zcr,
                          ll_rel, ll_abs, log_pow]
        # 5 + 1 + 2 + 1 + 3 = 12
    if TDA:
        feats = feats + _tda_features(sig)              # + 8 topological dims
    return feats


def channel_features(window: np.ndarray) -> np.ndarray:
    """window (C, T) -> (C, CHAN_FEAT_DIM) per-channel features."""
    out = np.array([_per_channel(window[c]) for c in range(window.shape[0])],
                   dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
