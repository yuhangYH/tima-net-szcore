# TIMA-Net for SzCORE

Docker image of the TIMA-Net seizure detector for the [SzCORE](https://epilepsybenchmarks.com) benchmark
(`esl-epfl/szcore`). It follows the SzCORE contract. The linux/amd64 image reads `/data/$INPUT` (EDF) and writes
a HED-SCORE / BIDS events TSV to `/output/$OUTPUT`.

```
docker run --rm --network none \
  -e INPUT=sub-01_ses-01_task-szMonitoring_run-00_eeg.edf \
  -e OUTPUT=sub-01_ses-01_task-szMonitoring_run-00_events.tsv \
  -v /path/to/edf:/data -v /path/to/out:/output mellow99/tima-net-szcore:v1-0-0
```

## Model

- One token per channel and 2-second window, made of 12 signal features plus a 4-D electrode-coordinate code
  (a bipolar pair uses both electrode positions). A 3-layer Transformer encoder (d_model 128, 4 heads) gives a
  window-level seizure probability.
- Five seeds trained on CHB-MIT, Siena, TUH-Sz (83 patients of TUSZ v2.0.3) and SeizeIT2. Epoch selection uses source
  validation patients only. The image averages the five seed probabilities.
- Post-processing (`algo/tima_szcore/szcore_config.json`) applies smoothing, a threshold, a merge gap, a minimum
  event duration and an optional per-record quantile gate. They were chosen once on the source validation
  patients converted to the SzCORE input format. Nothing was tuned on a SzCORE output.

## Input handling

- Channel names lose `EEG `/`POL ` prefixes, `-Avg`, `-REF`, `-LE`, `-AR`, `-CAR` suffixes and duplicate
  suffixes; T3/T4/T5/T6 map to T7/T8/P7/P8 (also inside bipolar names).
- If any of the 18 double-banana pairs is present the record is read as bipolar (CHB-MIT in SzCORE). Otherwise
  the 19 referential 10-20 electrodes present are used. Flat or nonfinite channels are omitted, never
  zero-filled. A record with no usable channel gets one `bckg` row.
- Signals are resampled to 256 Hz and tiled into contiguous 2-second windows from 0 s.

## Blind status

Dianalund and SeizeIT1 were never seen in training (SeizeIT1 shares a centre with the SeizeIT2 training source).
CHB-MIT, Siena and TUH are training corpora. The TUSZ patients used in training come from the `train`, `dev`
and `eval` splits, so subjects may overlap with the SzCORE TUH set.

## Layout

```
algo/tima_szcore/        __main__.py (entrypoint), core.py, model.py, channel_features.py, electrodes.py,
                         szcore_config.json, weights/seed{0..4}.pt
algo/requirements.txt
tima_net.yaml            SzCORE algorithm file
scripts/image_parity.py  compares the image output with reference probabilities
tests/                   entrypoint tests
```

`core.py`, `model.py`, `channel_features.py` and `electrodes.py` are byte-identical copies of the research
repository files, and a test there checks the hashes.

## License

MIT.
