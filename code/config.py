"""Where the released code looks for its inputs.

Every script imports its directories from here, so a fresh checkout runs without editing
constants.  Defaults are relative to the repository; override with environment variables
when the artefacts sit elsewhere:

    AUGUR_DATA    derived artefacts  (X.npy, Y.npy, M.npy, DUR.npy, vlen.npy,
                                      ds_meta.parquet, pred_*.npz, eta/)
    AUGUR_RAW     raw feed directory (metar_2025.parquet)
    AUGUR_TRACKS  terminal-area extracts (track_message_YYYYMM_clean.parquet)
    AUGUR_FIGS    figure output

The raw surveillance record is not redistributed; see README.md.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, os.pardir))


def _dir(var, *default):
    value = os.environ.get(var)
    return os.path.abspath(value) if value else os.path.join(REPO, *default)


DATA = _dir('AUGUR_DATA', 'data')
RAW = _dir('AUGUR_RAW', 'raw')
TRACKS = _dir('AUGUR_TRACKS', 'raw', 'tracks')
FIGS = _dir('AUGUR_FIGS', 'figs')


def read_parquet(path, **kw):
    """pandas.read_parquet with a fallback engine.

    The deposited parquet artefacts were written by the pyarrow of 2025.  pyarrow 19+
    refuses some of them with "Repetition level histogram size mismatch", a metadata check
    that has nothing to do with the values; fastparquet reads them unchanged.  Try the
    default engine first and fall back only on that failure, so a modern file is never
    routed through the slower reader.
    """
    import pandas as pd
    try:
        return pd.read_parquet(path, **kw)
    except OSError:
        return pd.read_parquet(path, engine='fastparquet', **kw)
