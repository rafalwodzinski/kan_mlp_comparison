"""
Shared publication-style helpers for all plotting modules.

- Larger fonts (readable after shrinking to a single IEEE column).
- Vector PDF output (sharp in LaTeX).
- Global Y-limit computation so that plots of one family are visually comparable.
"""

import os
from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

PUBLICATION_RC = {
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 15,
    "figure.titlesize": 18,
    "pdf.fonttype": 42,  # embed TrueType fonts (editable, IEEE-compliant)
    "ps.fonttype": 42,
}


def apply_publication_style(style: str = "whitegrid") -> None:
    """Applies seaborn theme and then the publication rcParams (seaborn resets rcParams)."""
    sns.set_theme(style=style, rc=PUBLICATION_RC)
    matplotlib.rcParams.update(PUBLICATION_RC)


def to_pdf_path(path: str) -> str:
    """Forces the .pdf extension on an output path."""
    root, _ = os.path.splitext(path)
    return root + ".pdf"


def save_pdf(path: str, fig: Optional[plt.Figure] = None) -> str:
    """Saves the (current) figure as vector PDF with tight bounding box. Returns the final path."""
    out_path = to_pdf_path(path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    (fig or plt.gcf()).savefig(out_path, format="pdf", bbox_inches="tight")
    return out_path


def global_ylim(*arrays: Iterable[float], pad: float = 0.05) -> Tuple[float, float]:
    """Returns (global_min, global_max) over all given arrays, padded by a fraction of the range."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    return lo - pad * span, hi + pad * span
