# data/

Empty in the repository. The derived artefacts are too large for git and are distributed
as one archive:

    augur-manoeuvre-v1.0.0.zip      Zenodo, DOI 10.5281/zenodo.XXXXXXX

Unpack it here so that `data/` holds, flat:

```
X.npy  M.npy  Y.npy  DUR.npy  vlen.npy
ds_meta.parquet  flight_times.parquet  ttl.parquet
arr_202501.parquet  arr_202502.parquet  arr_202503.parquet
pred_persistence.npz  pred_static.npz  pred_gru.npz  pred_tf.npz
pred_tcn_nostatic.npz  pred_tcn_single.npz  pred_tcn.npz  pred_tcn_s1.npz
eta/eta_test_preds.npz  eta/kine_s0.npz  eta/augur_s0.npz  eta/oracle_s0.npz
eta/shuffled_s0.npz
```

`X.npy` (85 MB) and `M.npy` (67 MB) are the input tensor and its mask; `Y.npy` and
`DUR.npy` are the censored targets; the three `arr_*.parquet` files (190 MB together) are
the terminal-area extracts from which `flights.py` rebuilds the 10 s tracks.

Checksums are in `../MANIFEST.md`. If the files sit elsewhere, set `AUGUR_DATA` instead of
moving them.

The raw surveillance record — `track_message_YYYYMM_clean.parquet` and
`metar_2025.parquet` — is not distributed. Steps 1–2 of the README need it; step 3 onward
does not.
