"""
OPUS file reader (Bruker Alpha II and compatible instruments).

Primary backend: brukeropus (pip install soilspec[opus]).
The reader returns a SpectraDataset so callers are backend-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np


def read_opus(
    paths: str | Path | Sequence[str | Path],
    block: str = "AB",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Read one or more Bruker OPUS files.

    Parameters
    ----------
    paths : str, Path, or list thereof
        File path(s) to OPUS files.
    block : str
        Spectral block to extract. Common values:
        ``'AB'`` absorbance (default), ``'SC'`` single channel,
        ``'R'`` reflectance.

    Returns
    -------
    X : (n_files, n_wavenumbers) float64
        Spectral matrix. All spectra are interpolated to the wavenumber
        grid of the first file when grids differ.
    wavenumbers : (n_wavenumbers,) float64
        Wavenumber axis in cm⁻¹ (descending, as OPUS stores them).
    sample_ids : list[str]
        Filenames (without extension) used as sample identifiers.

    Raises
    ------
    ImportError
        If brukeropus is not installed (install via ``pip install soilspec[opus]``).
    FileNotFoundError
        If any path does not exist.
    ValueError
        If the requested spectral block is not found in a file.
    """
    try:
        import brukeropus
    except ImportError as exc:
        raise ImportError(
            "brukeropus is required for OPUS file reading. "
            "Install it with:  pip install soilspec[opus]"
        ) from exc

    if isinstance(paths, (str, Path)):
        paths = [paths]

    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"OPUS file not found: {p}")

    spectra: list[np.ndarray] = []
    wavenumber_grids: list[np.ndarray] = []
    sample_ids: list[str] = []

    for p in paths:
        from brukeropus import read_opus as _read_opus
        opus_file = _read_opus(str(p))

        _block_attr = {'AB': 'a', 'SC': 'sm', 'RF': 'rf'}
        attr = _block_attr.get(block.upper())
        if attr is None or not hasattr(opus_file, attr):
            raise ValueError(f"Block '{block}' not found in {p.name}. Known mappings: {_block_attr}")
        block_data = getattr(opus_file, attr)

        wavenumber_grids.append(np.asarray(block_data.x, dtype=np.float64))
        spectra.append(np.asarray(block_data.y, dtype=np.float64))
        sample_ids.append(p.stem)

    # Align all spectra to the reference grid (first file)
    ref_wn = wavenumber_grids[0]
    aligned = []
    for wn, sp in zip(wavenumber_grids, spectra):
        if len(wn) == len(ref_wn) and np.allclose(wn, ref_wn, rtol=1e-4):
            aligned.append(sp)
        else:
            # Interpolate onto reference grid (np.interp needs ascending x)
            o = np.argsort(wn)
            aligned.append(np.interp(ref_wn, wn[o], sp[o]))

    X = np.vstack(aligned)
    return X, ref_wn, sample_ids
