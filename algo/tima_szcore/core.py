"""ideas-v2 SZD: TIMA-Net inference core for the SzCORE container (EDF -> window probabilities -> events).

Preregistration: ``docs/ideas_v2/PREREG_SZCORE_DEPLOY_20261118.md`` (sections 3 and 4).

This file is copied byte for byte into the Docker image as ``tima_szcore/core.py``; it only imports the three
modules that are copied with it (``model``, ``channel_features``, ``electrodes``) through relative imports, so
the same text runs as ``tima.v2i_szcore_infer`` in the repository and as ``tima_szcore.core`` in the image.

Signal path (identical to ``tima.phase1_data.load_native_audited`` + ``featurize_window`` for the original
corpora; checked on the HPC): MNE volts, no filtering, resample to 256 Hz, float32, recording-level omission of
channels that are flat or nonfinite anywhere, window-level omission of flat or nonfinite channels, the 12-D
Phase 1 feature per channel. Differences that only matter for unseen inputs: the SzCORE reference suffixes
(``-Avg`` and others) are stripped, the montage is detected from the channel names instead of the corpus name,
and a window without a valid channel or with nonfinite features gets probability 0 instead of stopping.
"""
from __future__ import annotations

from collections import defaultdict
import json
import re

import numpy as np

from . import channel_features as cf
from .electrodes import BIPOLAR_PAIRS, REFERENTIAL_1020, bipolar_geometry, referential_geometry

SFREQ = 256.0
WINDOW = 512                      # 2 s at 256 Hz
WINDOW_SEC = WINDOW / SFREQ
CHUNK_WINDOWS = 1800              # one hour of windows per read when no resampling is needed
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
BIPOLAR_NAMES = [f"{a}-{b}" for a, b in BIPOLAR_PAIRS]
REVERSED = {f"{b}-{a}": f"{a}-{b}" for a, b in BIPOLAR_PAIRS}
POSTPROCESS_KEYS = ("smooth", "threshold", "min_duration_sec", "merge_gap_sec", "gate_quantile")
GATE_MIN_SEC = 3600.0


def canonical_channel(name):
    """``phase1_data.canonical_channel`` plus the SzCORE/other reference suffixes ``-AVG``, ``-AR``, ``-CAR``."""
    name = re.sub(r"^(?:EEG|POL)\s*", "", str(name).strip().upper())
    name = name.replace(" ", "")
    name = re.sub(r"-\d+$", "", name)
    name = re.sub(r"-(?:REF|LE|AVG|AR|CAR)$", "", name)
    return "-".join(ALIASES.get(part, part) for part in name.split("-"))


def montage_of(canonical_names):
    """``"bipolar"`` if any of the 18 double-banana pairs (either direction) is a channel, else ``"referential"``."""
    names = set(canonical_names)
    return "bipolar" if names & (set(BIPOLAR_NAMES) | set(REVERSED)) else "referential"


def _bad(values):
    """Phase 1 QC reason for a 1-D signal: nonfinite anywhere, or exactly constant."""
    values = np.asarray(values)
    finite = np.isfinite(values)
    if not finite.all():
        return "nonfinite_signal"
    if values.size == 0:
        return "empty_signal"
    return "flat_signal" if values.max() == values.min() else None


class Recording:
    """Channel selection of one EDF; ``windows()`` yields ``(index, data[C, 512] float32)`` in order."""

    def __init__(self, edf_path, *, montage=None, common_average=False):
        import mne                                            # heavy import kept local
        self.audit = {"edf_path": str(edf_path), "warnings": []}
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        self.audit["channel_names"] = list(raw.ch_names)
        canonical = [canonical_channel(n) for n in raw.ch_names]
        self.montage = montage or montage_of(canonical)
        self.common_average = bool(common_average)
        if self.common_average and self.montage != "referential":
            raise ValueError("common-average conversion is defined for referential input only")
        acceptable = set(REFERENTIAL_1020)
        if self.montage == "bipolar":
            acceptable |= set(BIPOLAR_NAMES) | set(REVERSED)
        picks = [i for i, n in enumerate(canonical) if n in acceptable]
        self.audit.update(montage=self.montage, sfreq_in=float(raw.info["sfreq"]), n_times_in=int(raw.n_times),
                          ignored_channels=[raw.ch_names[i] for i in range(len(canonical)) if i not in picks])
        self.names, self.coords, self.n_windows = [], np.empty((0, 4), np.float32), 0
        if not picks:
            self.audit["exclusion_reason"] = "no_recognized_channels"
            raw.close()
            return
        raw.pick(picks)
        self._resample = abs(float(raw.info["sfreq"]) - SFREQ) > 1e-3
        # Recording-level QC on the raw channels, first valid occurrence of each canonical name.
        reasons = self._raw_reasons(raw)
        valid_by_name = defaultdict(list)
        for local, original in enumerate(picks):
            if reasons[local] is None:
                valid_by_name[canonical[original]].append(local)
        chosen = {name: locals_[0] for name, locals_ in valid_by_name.items()}
        if not chosen:
            self.audit["exclusion_reason"] = "no_valid_raw_channels"
            raw.close()
            return
        kept = sorted(set(chosen.values()))
        remap = {old: new for new, old in enumerate(kept)}
        raw.pick(kept)
        self.sources = {name: remap[local] for name, local in chosen.items()}
        expected = BIPOLAR_NAMES if self.montage == "bipolar" else list(REFERENTIAL_1020)
        geometry = bipolar_geometry() if self.montage == "bipolar" else referential_geometry(expected)
        self.plan = []                                        # (name, [(source index, sign)], geometry row)
        for index, name in enumerate(expected):
            if self.montage == "bipolar":
                a, b = name.split("-")
                if a in self.sources and b in self.sources:
                    terms = [(self.sources[a], 1.0), (self.sources[b], -1.0)]
                elif name in self.sources:
                    terms = [(self.sources[name], 1.0)]
                elif f"{b}-{a}" in self.sources:
                    terms = [(self.sources[f"{b}-{a}"], -1.0)]
                else:
                    continue
            elif name in self.sources:
                terms = [(self.sources[name], 1.0)]
            else:
                continue
            self.plan.append((name, terms, geometry[index]))
        self.raw = raw
        if self._resample:
            raw.load_data()
            raw.resample(SFREQ, verbose="ERROR")
            self._signals = raw.get_data()
            self.n_times = int(self._signals.shape[1])
        else:
            self._signals = None
            self.n_times = int(raw.n_times)
        self.n_windows = self.n_times // WINDOW
        # Derived-channel recording-level QC (Phase 1 checks the full derived float32 signal).
        c = len(self.plan)
        lo, hi, finite = np.full(c, np.inf), np.full(c, -np.inf), np.ones(c, bool)
        for start in range(0, self.n_times, CHUNK_WINDOWS * WINDOW):
            block = self._derive(start, min(self.n_times, start + CHUNK_WINDOWS * WINDOW), apply_car=False)
            finite &= np.isfinite(block).all(axis=1)
            safe = np.where(np.isfinite(block), block, 0.0)
            lo, hi = np.minimum(lo, safe.min(axis=1)), np.maximum(hi, safe.max(axis=1))
        keep = [i for i in range(c) if finite[i] and hi[i] != lo[i] and self.n_times > 0]
        self.audit["omitted_derived_channels"] = [self.plan[i][0] for i in range(len(self.plan)) if i not in keep]
        self.plan = [self.plan[i] for i in keep]
        self.names = [p[0] for p in self.plan]
        self.coords = np.asarray([p[2] for p in self.plan], np.float32).reshape(-1, 4)
        self.audit.update(channels=self.names, n_windows=self.n_windows, common_average=self.common_average)
        if not self.names:
            self.audit["exclusion_reason"] = "no_valid_derived_channels"

    @staticmethod
    def _raw_reasons(raw):
        """Phase 1 raw-channel QC (nonfinite anywhere, else exactly constant), one chunked pass."""
        n, c = int(raw.n_times), len(raw.ch_names)
        if n == 0:
            return ["empty_signal"] * c
        lo, hi = np.full(c, np.inf), np.full(c, -np.inf)
        finite = np.ones(c, bool)
        step = CHUNK_WINDOWS * 2 * int(round(float(raw.info["sfreq"])))
        for start in range(0, n, step):
            values = raw.get_data(start=start, stop=min(n, start + step))
            ok = np.isfinite(values).all(axis=1)
            finite &= ok
            safe = np.where(np.isfinite(values), values, 0.0)
            lo, hi = np.minimum(lo, safe.min(axis=1)), np.maximum(hi, safe.max(axis=1))
        return [None if f and h != l else ("nonfinite_signal" if not f else "flat_signal")
                for f, l, h in zip(finite, lo, hi)]

    def _source_block(self, start, stop):
        if self._signals is not None:
            return self._signals[:, start:stop]
        return self.raw.get_data(start=start, stop=stop)

    def _derive(self, start, stop, apply_car=True):
        """Derived float32 channels of samples ``[start, stop)`` (float64 arithmetic before the cast, as Phase 1)."""
        src = self._source_block(start, stop)
        if apply_car and self.common_average:
            # SzCORE format emulation for referential corpora: subtract the mean of the electrodes kept.
            idx = [terms[0][0] for _, terms, _ in self.plan]
            src = src.copy()
            src[idx] = src[idx] - src[idx].mean(axis=0, keepdims=True)
        out = np.empty((len(self.plan), stop - start), np.float32)
        for i, (_, terms, _) in enumerate(self.plan):
            if len(terms) == 2:
                signal = src[terms[0][0]] - src[terms[1][0]]
            elif terms[0][1] < 0:
                signal = -src[terms[0][0]]
            else:
                signal = src[terms[0][0]]
            out[i] = np.asarray(signal, dtype=np.float32)
        return out

    def windows(self):
        if not self.names:
            return
        for first in range(0, self.n_windows, CHUNK_WINDOWS):
            last = min(self.n_windows, first + CHUNK_WINDOWS)
            block = self._derive(first * WINDOW, last * WINDOW)
            for k in range(last - first):
                yield first + k, block[:, k * WINDOW:(k + 1) * WINDOW]

    def close(self):
        raw = getattr(self, "raw", None)
        if raw is not None and hasattr(raw, "close"):
            raw.close()


def featurize(window, coords, names):
    """``phase1_data.featurize_window`` without its exception: ``(feats, coords)`` or ``None`` for no token."""
    if cf.LEGACY9 or cf.TDA or cf.CHAN_FEAT_DIM != 12:
        raise ValueError("TIMA SzCORE requires the default 12-D features; unset TIMA_FEAT_LEGACY9 / TIMA_FEAT_TDA")
    window = np.asarray(window, np.float32)
    kept = [i for i in range(len(names)) if _bad(window[i]) is None]
    if not kept:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        feats = np.asarray([cf._per_channel(window[i]) for i in kept], dtype=np.float32)
    if feats.shape != (len(kept), 12) or not np.isfinite(feats).all():
        return None
    return feats, np.asarray(coords, np.float32)[kept]


def recording_features(recording):
    """List of ``(window index, feats, coords)`` for windows with a valid token; plus the skipped indices."""
    items, skipped = [], []
    for index, window in recording.windows():
        out = featurize(window, recording.coords, recording.names)
        if out is None:
            skipped.append(index)
        else:
            items.append((index, out[0], out[1]))
    return items, skipped


def load_models(checkpoints, device="cpu"):
    import torch
    from .model import TIMANet
    models = []
    for path in checkpoints:
        payload = torch.load(str(path), map_location=device, weights_only=False)
        if payload.get("model_class") != "tima.model.TIMANet" or payload.get("scaler", {}).get("type", "identity") != "identity":
            raise ValueError(f"unexpected checkpoint {path}")
        model = TIMANet(**payload["model_kwargs"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models


def predict_items(models, items, *, batch_size=256, device="cpu"):
    """Mean over models of the softmax seizure probability, one value per item (Phase 1 padding/masking)."""
    import torch
    probs = np.zeros((len(models), len(items)), np.float64)
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            width = max(f.shape[0] for _, f, _ in batch)
            feats = torch.zeros(len(batch), width, 12)
            coords = torch.zeros(len(batch), width, 4)
            mask = torch.ones(len(batch), width, dtype=torch.bool)
            for j, (_, f, c) in enumerate(batch):
                feats[j, :len(f)] = torch.from_numpy(f)
                coords[j, :len(c)] = torch.from_numpy(c)
                mask[j, :len(f)] = False
            for m, model in enumerate(models):
                logits = model(feats.to(device), coords.to(device), key_padding_mask=mask.to(device))
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite inference logits")
                probs[m, start:start + len(batch)] = torch.softmax(logits, dim=-1)[:, 1].double().cpu().numpy()
    return probs.mean(axis=0), probs


def window_probabilities(models, recording, *, batch_size=256, device="cpu"):
    """``(p[n_windows], per_model[M, n_windows], skipped indices)``; skipped windows get 0."""
    items, skipped = recording_features(recording)
    p = np.zeros(recording.n_windows, np.float64)
    per = np.zeros((len(models), recording.n_windows), np.float64)
    if items:
        mean, each = predict_items(models, items, batch_size=batch_size, device=device)
        index = np.asarray([i for i, _, _ in items], dtype=np.int64)
        p[index], per[:, index] = mean, each
    return p, per, skipped


def smooth(p, k):
    """Centred moving average over ``k`` windows (odd); the mean over the windows present at the edges."""
    p = np.asarray(p, np.float64)
    if k <= 1 or len(p) == 0:
        return p.copy()
    half = k // 2
    c = np.r_[0.0, np.cumsum(p)]
    lo = np.clip(np.arange(len(p)) - half, 0, len(p))
    hi = np.clip(np.arange(len(p)) + half + 1, 0, len(p))
    return (c[hi] - c[lo]) / (hi - lo)


def events_from_probabilities(p, config, window_sec=WINDOW_SEC):
    """Section 4 rule: smooth, threshold (optionally raised by the record quantile gate), merge, minimum duration.

    Returns a list of ``(onset_sec, offset_sec)``; events are runs of above-threshold windows, gaps of at most
    ``merge_gap_sec`` are merged first, then events shorter than ``min_duration_sec`` are dropped.
    """
    config = {k: config[k] for k in POSTPROCESS_KEYS}
    s = smooth(p, int(config["smooth"]))
    threshold = float(config["threshold"])
    if config["gate_quantile"] is not None and len(s) * window_sec >= GATE_MIN_SEC:
        threshold = max(threshold, float(np.quantile(s, float(config["gate_quantile"]))))
    above = s >= threshold
    if not above.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(np.int8), 0]))
    runs = [[a * window_sec, b * window_sec] for a, b in zip(edges[::2], edges[1::2])]
    merged = [runs[0]]
    for onset, offset in runs[1:]:
        if onset - merged[-1][1] <= float(config["merge_gap_sec"]):
            merged[-1][1] = offset
        else:
            merged.append([onset, offset])
    return [(float(a), float(b)) for a, b in merged if b - a >= float(config["min_duration_sec"])]


def write_tsv(out_path, events, *, duration_sec, date_time="n/a"):
    """SzCORE / BIDS events TSV (columns as the MV-AFA submission); one ``bckg`` row when there is no event."""
    rows = [(onset, offset - onset, "sz") for onset, offset in events] or [(0.0, float(duration_sec), "bckg")]
    lines = ["onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration"]
    lines += [f"{o}\t{d}\t{t}\tn/a\tn/a\t{date_time}\t{float(duration_sec)}" for o, d, t in rows]
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(rows)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    missing = [k for k in POSTPROCESS_KEYS if k not in config["postprocess"]]
    if missing:
        raise ValueError(f"postprocess config misses {missing}")
    return config
