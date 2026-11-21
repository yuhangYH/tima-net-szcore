"""Write a SzCORE-format copy of a referential EDF: the 19 10-20 electrodes, common average, ``<Elec>-Avg`` names.

    python scripts/make_avg_copy.py input.edf output.edf

Mirrors the in-memory conversion of the tuning panel (TIMA ``common_average=True``) so the image output on the copy
can be compared with the panel probabilities. Needs mne and edfio. EDF re-quantisation adds small differences.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algo"))
from tima_szcore import core  # noqa: E402
from tima_szcore.electrodes import REFERENTIAL_1020  # noqa: E402

OLD = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def main(src, dst):
    import mne
    raw = mne.io.read_raw_edf(src, preload=True, verbose="ERROR")
    canon = [core.canonical_channel(n) for n in raw.ch_names]
    picks, names = [], []
    for elec in REFERENTIAL_1020:
        if elec in canon:
            picks.append(canon.index(elec))
            names.append(f"{OLD.get(elec, elec).title()}-Avg")
    data = raw.get_data(picks=picks)
    data = data - data.mean(axis=0, keepdims=True)
    info = mne.create_info(names, raw.info["sfreq"], "eeg")
    out = mne.io.RawArray(data, info, verbose="ERROR")
    lim = float(np.abs(data).max()) * 1.01 * 1e6      # mne writes EEG in uV; physical_range is in uV, not V
    out.export(dst, fmt="edf", physical_range=(-lim, lim), overwrite=True, verbose="ERROR")
    back = mne.io.read_raw_edf(dst, preload=True, verbose="ERROR").get_data()[:, :data.shape[1]]
    err = float(np.abs(back - data).max())
    if err > 2 * lim * 1e-6 / 65535:                   # more than one 16-bit EDF step: the copy is wrong
        raise SystemExit(f"read-back error {err:.3g} V exceeds one quantisation step")
    print(f"read-back max abs error {err:.3g} V (16-bit step {2 * lim * 1e-6 / 65535:.3g} V)")
    print(f"{len(names)} channels, {data.shape[1] / raw.info['sfreq']:.1f} s -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
