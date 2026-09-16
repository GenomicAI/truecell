"""Generate all figures for the integration tutorial.

Runs ifnb_integration_tutorial.run_full(do_umap=True) and renders the
truecell-side figures to tutorials/figures_integration/, alongside the r_*.png the
R verify script writes:
  * py_01_uncorrected_stim.png   UMAP of raw PCA, coloured by condition (the batch)
  * py_02_harmony_stim.png       UMAP after Harmony, by condition (now mixed)
  * py_03_harmony_celltype.png   UMAP after Harmony, by cell type (still separated)
  * py_04_scoreboard.png         batch separation vs cell-type recovery, per method

Usage
-----
    python tutorials/generate_integration_plots.py [--data-dir PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tutorials.ifnb_integration_tutorial import run_full, BATCH, CELLTYPE
from truecell.plotting import dim_plot, _palette

FIGURES = Path(__file__).parent / "figures_integration"
FIGURES.mkdir(exist_ok=True)


def _save(fig, name):
    import matplotlib.pyplot as plt

    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


def scoreboard_bars(summary):
    """Two grouped bars per method: batch separation (mix) and cell-type recovery.

    The story of integration in one panel — every method should push the left
    bar (batch silhouette, want low) down and hold the right bar (cell-type ARI,
    want high) up relative to the uncorrected baseline.

    Laid out so that nothing can land on a bar whatever the scores are. The
    manuscript printed this at 0.55 of its size, where the legend, placed
    wherever matplotlib found room, sat on the bars. The legend now sits under
    the method names, and constrained layout keeps room for it at any size;
    ``tight_layout`` measures once, at the size the figure was made. The axis
    runs past 1, the most either score can reach, so the value label on the
    tallest possible bar still fits below the title.
    ``tests/test_integration_scoreboard.py`` checks both sizes.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    board = summary["scoreboard"]
    methods = list(board["method"])
    sil_batch = board["sil_batch"].to_numpy(dtype=float)
    ari = board["ari_celltype"].to_numpy(dtype=float)

    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    # Two lines each, so the legend fits across the figure at the manuscript's size.
    b1 = ax.bar(x - w / 2, sil_batch, w, label="batch separation\n(sil_batch ↓ better)",
                color=_palette(3)[2])
    b2 = ax.bar(x + w / 2, ari, w, label="cell-type recovery\n(ARI ↑ better)",
                color=_palette(3)[0])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("score")
    # Silhouettes and ARI can go below zero; a negative bar's label hangs below it.
    lowest = min(0.0, float(np.nanmin(np.concatenate([sil_batch, ari]))))
    ax.set_ylim(lowest - (0.15 if lowest < 0 else 0.0), 1.15)
    ax.set_title("Integration scoreboard\nbatch mixing vs cell-type preservation")
    fig.legend(fontsize=8, frameon=False, loc="outside lower center", ncol=2)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    return fig


def main(data_dir=None):
    obj, summary = run_full(data_dir=data_dir, verbose=False, do_umap=True)

    # 1. Uncorrected: the two conditions form two clouds — the batch effect.
    _save(dim_plot(obj, reduction="umap_pca", group_by=BATCH, label=False,
                   pt_size=2, title="Uncorrected (PCA) — by condition"),
          "py_01_uncorrected_stim.png")

    # 2. After Harmony: the conditions overlap — batches mixed.
    _save(dim_plot(obj, reduction="umap_harmony", group_by=BATCH, label=False,
                   pt_size=2, title="Harmony — by condition"),
          "py_02_harmony_stim.png")

    # 3. Same map by cell type: the biology survived the correction.
    _save(dim_plot(obj, reduction="umap_harmony", group_by=CELLTYPE, label=True, repel=True,
                   pt_size=2, label_size=7, title="Harmony — by cell type"),
          "py_03_harmony_celltype.png")

    # 4. The headline: every method's batch-vs-biology trade-off, side by side.
    _save(scoreboard_bars(summary), "py_04_scoreboard.png")

    print(f"\n  Wrote figures to {FIGURES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ifnb integration tutorial figures")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    main(data_dir=args.data_dir)
