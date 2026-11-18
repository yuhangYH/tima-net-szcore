"""Entrypoint tests on synthetic EDFs (needs mne, torch and edfio). Run: python -m unittest discover tests"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algo"))

from tima_szcore import core  # noqa: E402
from tima_szcore.electrodes import REFERENTIAL_1020  # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "algo" / "tima_szcore"
HAVE_WEIGHTS = len(list((PKG / "weights").glob("seed*.pt"))) >= 3 and (PKG / "szcore_config.json").exists()


def write_edf(path, names, data, sfreq):
    import mne
    info = mne.create_info(names, sfreq, "eeg")
    mne.io.RawArray(data, info, verbose="ERROR").export(str(path), fmt="edf", physical_range=(-1e-3, 1e-3),
                                                       overwrite=True, verbose="ERROR")


@unittest.skipUnless(HAVE_WEIGHTS, "weights not copied yet")
class EntrypointTests(unittest.TestCase):
    def _run(self, names, sfreq=256.0, seconds=60):
        from tima_szcore.__main__ import main
        rng = np.random.default_rng(0)
        data = rng.normal(scale=2e-5, size=(len(names), int(sfreq * seconds)))
        with tempfile.TemporaryDirectory() as tmp:
            edf, tsv, probs = Path(tmp) / "x.edf", Path(tmp) / "o" / "x.tsv", Path(tmp) / "p.npy"
            write_edf(edf, names, data, sfreq)
            main([str(edf), str(tsv), "--probs", str(probs)])
            return tsv.read_text().splitlines(), np.load(probs)

    def test_config_has_postprocess(self):
        cfg = core.load_config(PKG / "szcore_config.json")
        self.assertEqual(set(core.POSTPROCESS_KEYS), set(cfg["postprocess"]))

    def test_avg_referential(self):
        names = [f"{'T3' if e == 'T7' else e.title()}-Avg" for e in REFERENTIAL_1020]
        lines, p = self._run(names, sfreq=250.0)
        self.assertEqual(lines[0].split("\t")[0], "onset")
        self.assertEqual(len(p), 30)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        self.assertAlmostEqual(float(lines[1].split("\t")[-1]), 60.0, places=3)

    def test_unknown_channels_give_bckg(self):
        lines, p = self._run(["ECG", "EMG"])
        self.assertEqual(len(p), 0)
        self.assertEqual(lines[1].split("\t")[2], "bckg")


if __name__ == "__main__":
    unittest.main()
