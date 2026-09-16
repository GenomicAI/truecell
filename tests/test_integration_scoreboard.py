"""The integration scoreboard keeps its legend and value labels off its bars.

Reviewer 3 found them on the bars and against the title where the manuscript
printed this figure. The layout is checked at the tutorial's size and at the 0.55
the manuscript prints it at, on the tutorial's own scores and on scores chosen to
push the layout: a perfect ARI, negative silhouettes, a zero-height bar, and six
methods rather than four.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import _layout

from tutorials.generate_integration_plots import scoreboard_bars

BOARDS = {
    # integration_vignette.md's table
    "vignette": pd.DataFrame({
        "method": ["uncorrected (PCA)", "harmony", "cca", "rpca"],
        "sil_batch": [0.1070, 0.0077, 0.0039, 0.0052],
        "ari_celltype": [0.5263, 0.9217, 0.9280, 0.7300]}),
    "edges": pd.DataFrame({
        "method": ["uncorrected (PCA)", "harmony", "cca", "rpca", "fastmnn", "scvi"],
        "sil_batch": [0.35, -0.08, 0.0, 0.004, 1.0, -0.2],
        "ari_celltype": [1.0, 0.0, 0.93, 0.999, 0.5, 1.0]}),
}


@pytest.mark.parametrize("name", sorted(BOARDS))
@pytest.mark.parametrize("shrink", [False, True])
def test_the_scoreboard_is_laid_out_cleanly(name, shrink):
    fig = scoreboard_bars({"scoreboard": BOARDS[name]})
    try:
        if shrink:
            _layout.shrink(fig)
        assert _layout.check_layout(fig) == []
    finally:
        plt.close(fig)


def test_every_score_is_labelled_on_its_bar():
    board = BOARDS["edges"]
    fig = scoreboard_bars({"scoreboard": board})
    try:
        labels = [t.get_text() for t in fig.axes[0].texts]
        expected = [f"{v:.2f}" for v in board["sil_batch"]] + [f"{v:.2f}" for v in board["ari_celltype"]]
        assert labels == expected
    finally:
        plt.close(fig)
