# AUGUR — manoeuvre-event forecasting in terminal airspace

Code and derived data for

> *Forecasting trajectory-observable manoeuvre events in terminal airspace: a date-split
> protocol on measured surveillance data*

The paper changes the prediction target from arrival time to a manoeuvre that is visible in
the surveillance track — close-in vectoring (`VEC`), level-off (`LVL`), speed control
(`SPD`) — and scores it as an event, not as a frame. This repository holds the code that
produced every number in the paper, from the raw feed to the tables and figures.

The raw surveillance record belongs to the air navigation service provider and is not
redistributed. Everything derived from it is: see **Data** below.

## Contents

```
code/            pipeline, appendix A, appendix B, figures
data/            derived artefacts (downloaded, not versioned)
raw/             raw feed (not redistributed; only needed for the first two steps)
figs/            figure output
```

| file | what it does | paper |
|---|---|---|
| `build_tracks.py` | streams the monthly feed, keeps 70 min of track before touchdown | §3.1, Table 1 |
| `flights.py` | resamples each arrival on a 10 s grid in the runway-anchored frame | §3.1, Fig. 2 |
| `build_ds.py` | 60 km anchor, 60-frame input, censored targets, date splits | §3.3, §5.1, Table 3 |
| `build_ttl.py` | time from `t0` to the end of the contiguous track | Appendix B |
| `labels.py` | the three manoeuvre definitions and the `HOLD > VEC > LVL > SPD` precedence | §4.2 |
| `model.py` | dilated causal encoder, occupancy / onset / duration heads | §5.2, Fig. 5 |
| `train.py` | one configuration, probabilities for validation and test | §5.4 |
| `baselines.py` | no-event floor and static logistic reference | §5.3, Table 4 |
| `metrics.py` | event extraction and one-to-one event matching | §6.1 |
| `aggregate.py` | day-block bootstrap, paired differences | §6.3, §7, Tables 4–6 |
| `sensitivity.py`, `tol_sens.py`, `spd_alt.py` | definition and tolerance sweeps, SPD scale | Appendix A, Tables 7–9 |
| `eta_experiment.py`, `eta_figs.py` | arrival-time cross-check | Appendix B, Table 10 |
| `figs_data.py`, `figs_results.py` | figures 2–12 | §3–§7 |
| `config.py` | directories, and a parquet reader with a fallback engine | — |

## Requirements

Python 3.11 with the packages in `requirements.txt`. Training ran on CPU; no GPU is
required, and none of the reported numbers depends on one.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
```

`config.py` reads four directories from the environment, defaulting to this repository:

| variable | default | holds |
|---|---|---|
| `AUGUR_DATA` | `data/` | `X.npy`, `Y.npy`, `M.npy`, `DUR.npy`, `vlen.npy`, `ds_meta.parquet`, `pred_*.npz`, `eta/` |
| `AUGUR_RAW` | `raw/` | `metar_2025.parquet` |
| `AUGUR_TRACKS` | `raw/tracks/` | `track_message_YYYYMM_clean.parquet` |
| `AUGUR_FIGS` | `figs/` | figure output |

## Data

The derived artefacts are too large for git and are deposited as one archive,
`augur-manoeuvre-v1.0.0.zip` (DOI `10.5281/zenodo.XXXXXXX`). It holds this whole tree with
`data/` filled in, so either unpack it and run, or — if you cloned from GitHub — unpack it
into `data/`. Either way `data/` ends up flat, as listed in `data/README.md`. For 26,592
arrivals and the 90-day study period it contains:

| file | contents |
|---|---|
| `X.npy`, `M.npy` | input tensor and its valid-frame mask |
| `Y.npy`, `DUR.npy` | censored occupancy targets and event durations |
| `vlen.npy` | valid length of each arrival's future |
| `ds_meta.parquet` | flight, day, decision instant `t0`, valid length; the split is assigned from `day` by `data.SPLITS` |
| `arr_2025{01,02,03}.parquet` | terminal-area extracts, one row per surveillance report |
| `flight_times.parquet`, `ttl.parquet` | track end times and time-to-land |
| `pred_*.npz` | per-arrival test probabilities for every method in Table 4 |
| `eta/` | predictions behind Table 10 |

Steps 1–2 need the raw feed and are only reproducible by the data owner; they are included
because the extraction is part of the claim (Table 1), not because the deposit depends on
them. Steps 3 onward run from the deposit alone.

## Reproducing

Run from `code/`. Each command writes into `AUGUR_DATA`.

```
# 1  raw feed -> terminal-area extracts                    (needs AUGUR_TRACKS)
python build_tracks.py

# 2  extracts -> 10 s resampled tracks and per-flight metadata
python flights.py

# 3  resampled tracks -> tensors, targets, splits           (needs AUGUR_RAW for METAR)
python build_ds.py

# 4  time-to-land per arrival
python build_ttl.py

# 5  the reported model, 22 epochs, about 17 min on CPU
python train.py --tag tcn --epochs 22

# 6  references: GRU, Transformer, static logistic, no event
python train.py --enc gru --tag gru --epochs 22
python train.py --enc tf  --tag tf  --epochs 22
python baselines.py
python train.py --tag tcn_nostatic --no_static --epochs 22
python train.py --tag tcn_single --single_task --epochs 22
python train.py --tag tcn_s1 --seed 1 --epochs 22

# 7  event-level scores, day-block intervals, paired differences   -> Tables 4-6
python aggregate.py

# 8  appendix A                                                     -> Tables 7-9
python sensitivity.py
python tol_sens.py
python spd_alt.py

# 9  appendix B                                                     -> Table 10
python eta_experiment.py agg
python eta_figs.py

# 10 figures
python figs_data.py
python figs_results.py
```

`aggregate.py` prints macro F1 for every method and writes `results_test.json`,
`results_test.csv` and `paired_diff.json`. On the deposited probabilities it prints
0.599 for `tcn`, 0.900 / 0.633 / 0.262 by type, 0.346 for `static` and 0.000 for
`persistence`, which are the numbers in the abstract.

## Notes on the deposit

- The split is not a column in `ds_meta.parquet`. It follows from `day` and the three
  calendar constants in `data.SPLITS`, so the assignment cannot drift from the code that
  scores it; `python -c "import data; print(data.SPLITS)"` prints the boundaries.
- The parquet artefacts were written by the pyarrow of 2025. pyarrow 19+ rejects some of
  them with `Repetition level histogram size mismatch`, a metadata check unrelated to the
  values. `config.read_parquet` falls back to `fastparquet` on exactly that error, so
  neither pandas nor the reader needs pinning.
- The operating threshold is selected on the validation split inside `aggregate.py`, under
  the same admissibility used for reporting, rather than read from training. This is why
  rerunning `aggregate.py` reproduces the published scores exactly.
- `pred_persistence.npz` is the no-event reference: an all-zero probability tensor, kept in
  the same format as the learned methods so that one code path scores everything.

## Licence

Code under MIT (`LICENSE`); derived artefacts under CC BY 4.0 (`LICENSE-DATA`).

## Citing

See `CITATION.cff`.
