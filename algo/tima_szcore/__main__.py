"""SzCORE entrypoint: ``python -m tima_szcore <input.edf> <output.tsv> [--probs probs.npy]``.

Inside the container SzCORE runs ``python3 -m tima_szcore "/data/${INPUT}" "/output/${OUTPUT}"``.
``core.py``, ``model.py``, ``channel_features.py`` and ``electrodes.py`` are byte-identical copies of the TIMA-Net
research repository files (``tima/v2i_szcore_infer.py`` and the three modules); the five seed checkpoints and the
post-processing setting (``szcore_config.json``) were fixed on source-validation patients only.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import __version__, core

HERE = Path(__file__).resolve().parent


def _log(message):
    print(f"[tima_szcore] {message}", file=sys.stderr, flush=True)


def _start_datetime(edf_path):
    import mne
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        meas = raw.info.get("meas_date")
        raw.close()
        return meas.replace(tzinfo=None).isoformat() if meas is not None else "n/a"
    except Exception:                                   # dateTime is informational only
        return "n/a"


def main(argv=None):
    ap = argparse.ArgumentParser(description="TIMA-Net seizure detection (SzCORE)")
    ap.add_argument("input", help="input EDF")
    ap.add_argument("output", help="output HED-SCORE / BIDS events TSV")
    ap.add_argument("--probs", help="optional .npy for the per-window ensemble probabilities (parity checks)")
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    config = core.load_config(HERE / "szcore_config.json")
    checkpoints = sorted((HERE / "weights").glob("seed*.pt"))
    if len(checkpoints) < 3:
        raise RuntimeError(f"expected the seed checkpoints in {HERE / 'weights'}")
    _log(f"version {__version__}, {len(checkpoints)} models, postprocess {json.dumps(config['postprocess'])}")
    models = core.load_models(checkpoints)
    rec = core.Recording(args.input)
    duration = rec.audit["n_times_in"] / rec.audit["sfreq_in"]
    try:
        p, _, skipped = core.window_probabilities(models, rec)
    finally:
        rec.close()
    _log(f"montage={rec.audit.get('montage')} channels={len(rec.names)} windows={len(p)} skipped={len(skipped)} "
         f"exclusion={rec.audit.get('exclusion_reason')} ignored={rec.audit.get('ignored_channels')}")
    events = core.events_from_probabilities(p, config["postprocess"]) if len(p) else []
    events = [(a, min(b, duration)) for a, b in events if a < duration]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = core.write_tsv(args.output, events, duration_sec=duration, date_time=_start_datetime(args.input))
    if args.probs:
        np.save(args.probs, p)
    _log(f"{len(events)} event(s), {rows} row(s) -> {args.output}")


if __name__ == "__main__":
    main()
