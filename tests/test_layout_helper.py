"""The checks in ``tests/_layout.py`` find what they say they find.

A layout check that can not fail proves nothing about a figure that passes it, so
each kind of problem is built here on purpose, next to the near miss the check
has to let through.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _layout


@pytest.fixture
def new_figure():
    made = []

    def make(**kwargs):
        fig, ax = plt.subplots(**kwargs)
        made.append(fig)
        return fig, ax

    yield make
    for fig in made:
        plt.close(fig)


def test_text_over_text_is_found(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    ax.text(0.5, 0.50, "first label", ha="center", va="center")
    ax.text(0.5, 0.53, "second label", ha="center", va="center")
    assert _layout.check_layout(fig) == ["text overlaps text: 'first label' / 'second label'"]
    assert _layout.check_layout(fig, allow={("second label", "first label")}) == []


def test_text_that_does_not_touch_is_clean(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    ax.text(0.5, 0.3, "first label", ha="center", va="center")
    ax.text(0.5, 0.7, "second label", ha="center", va="center")
    assert _layout.check_layout(fig) == []


def test_the_box_behind_a_label_counts_as_part_of_it(new_figure):
    """Two labels whose white boxes overlap hide each other's edges, even when
    the letters themselves are a pixel or two apart."""
    fig, ax = new_figure(figsize=(4, 3))
    upper = ax.text(0.5, 0.5, "upper", ha="center", va="bottom")
    lower = ax.text(0.5, 0.5, "lower", ha="center", va="top")
    assert _layout.check_layout(fig) == [], "premise: the letters only meet"
    for text in (upper, lower):
        text.set_bbox({"boxstyle": "round,pad=0.6", "fc": "white"})
    assert _layout.check_layout(fig) == ["text overlaps text: 'upper' / 'lower'"]


def test_rotated_tick_labels_are_compared_as_turned_rectangles(new_figure):
    """Under a violin plot the group names run at 45 degrees. Their upright
    bounding boxes overlap one another while the text itself does not."""
    names = ["CD14+ Mono", "FCGR3A+ Mono", "Memory CD4 T", "Naive CD4 T",
             "Plasmacytoid DC", "Megakaryocyte", "Erythrocyte", "Platelet"]
    fig, ax = new_figure(figsize=(3.5, 3.5))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(names) - 0.5)
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    upright = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    assert any(_layout._overlap(a, b) for a, b in itertools.pairwise(upright)), \
        "premise: the upright boxes overlap by the check's own measure"
    assert _layout.check_layout(fig, tight_bbox=True) == []

    ax.set_xticklabels(names, rotation=0, ha="center")
    fig.set_size_inches(3, 3)
    assert any(p.startswith("text overlaps text") for p in _layout.check_layout(fig, tight_bbox=True))


def test_text_off_the_canvas_is_found_unless_the_save_grows_the_canvas(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    ax.text(1.3, 0.5, "outside", transform=ax.transAxes)
    assert _layout.check_layout(fig) == ["text off the canvas: 'outside'"]
    assert _layout.check_layout(fig, tight_bbox=True) == []


def test_tick_labels_matplotlib_does_not_draw_are_ignored(new_figure):
    """Ticks just outside the view interval keep visible labels that are never
    drawn. Listed naively, a UMAP's ``−10`` read as text off the canvas."""
    fig, ax = new_figure(figsize=(4, 3))
    ax.scatter([0, 1, 2], [0, 1, 2])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lo, hi = ax.get_xlim()
    undrawn = [label for tick, label in zip(ax.get_xticks(), ax.get_xticklabels())
               if not lo <= tick <= hi and label.get_visible() and label.get_text()]
    assert any(label.get_window_extent(renderer).x0 < 0 for label in undrawn), \
        "premise: an undrawn tick label sits off the canvas"
    assert _layout.check_layout(fig) == []


def test_a_legend_over_markers_is_found(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    ax.scatter(np.linspace(0, 1, 40), np.full(40, 0.95), label="along the top")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    assert any(p.startswith("legend covers") and "marker" in p for p in _layout.check_layout(fig))
    ax.legend(loc="lower right")
    assert _layout.check_layout(fig) == []


def test_a_legend_over_visium_spots_is_found(new_figure):
    """``spatial_dim_plot`` draws its spots as an EllipseCollection, not scatter
    markers, so the check has to know about both."""
    from matplotlib.collections import EllipseCollection

    fig, ax = new_figure(figsize=(4, 3))
    xy = np.column_stack([np.linspace(0.05, 0.95, 30), np.full(30, 0.9)])
    ax.add_collection(EllipseCollection(0.03, 0.03, 0, units="x", offsets=xy,
                                        offset_transform=ax.transData, label="spots"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([], [], "o", label="a group")
    ax.legend(loc="upper left")
    assert any(p.startswith("legend covers") for p in _layout.check_layout(fig))
    ax.legend(loc="lower right")
    assert _layout.check_layout(fig) == []


def test_a_figure_legend_over_markers_is_found(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    ax.scatter(np.linspace(0, 1, 40), np.linspace(0, 1, 40), label="diagonal")
    fig.legend(loc="center")
    assert any(p.startswith("legend covers") for p in _layout.check_layout(fig))


def test_markers_clipped_out_of_view_can_not_be_covered(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    ax.scatter(np.linspace(0, 3, 60), np.linspace(0, 1, 60), label="runs past the limit")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    assert _layout.check_layout(fig, tight_bbox=True) == []


def test_a_legend_over_bars_is_found(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    ax.bar([0, 1, 2], [1.0, 1.0, 1.0], label="bars")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center")
    assert any(p.startswith("legend covers") and "bar(s)" in p for p in _layout.check_layout(fig))
    ax.set_ylim(0, 2.0)
    assert _layout.check_layout(fig) == []


def test_a_value_label_on_a_neighbouring_bar_is_found(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    bars = ax.bar([0, 1], [0.05, 1.0], width=0.9)
    ax.bar_label(bars, fmt="%.2f", padding=2)
    assert _layout.check_layout(fig) == []
    ax.text(0.5, 0.5, "a long note placed over the bars")
    assert "annotation overlaps 1 bar(s): 'a long note placed over the bars'" in _layout.check_layout(fig)


def test_an_annotation_over_a_legend_is_found(new_figure):
    fig, ax = new_figure(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label="a line")
    legend = ax.legend(loc="upper left")
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    x, y = ax.transData.inverted().transform(((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2))
    ax.text(x, y, "note", ha="center", va="center")
    assert "annotation overlaps a legend: 'note'" in _layout.check_layout(fig)


def test_a_label_on_the_data_may_cover_markers_but_not_hide_a_group(new_figure):
    rng = np.random.default_rng(0)
    fig, ax = new_figure(figsize=(4, 3))
    ax.scatter(rng.normal(0, 1, 400), rng.normal(0, 1, 400), s=4, label="big")
    ax.scatter(rng.normal(4, 0.02, 12), rng.normal(0, 0.02, 12), s=4, label="small")
    ax.text(0, 0, "big", ha="center", va="center")
    ax.text(4, 0, "small", ha="center", va="center")

    plain = _layout.check_layout(fig)
    assert any(p.startswith("annotation overlaps") and p.endswith("'big'") for p in plain)
    assert any(p.startswith("annotation overlaps") and p.endswith("'small'") for p in plain)

    assert _layout.check_layout(fig, on_data={"big", "small"}) == [
        "label 'small' hides 12 of 12 markers of 'small'"]


def test_smallest_type_scales_with_the_width_the_figure_is_printed_at(new_figure):
    fig, ax = new_figure(figsize=(7, 3))
    ax.set_title("title", fontsize=12)
    ax.text(0.5, 0.5, "note", fontsize=6)
    assert _layout.smallest_type_pt(fig) == pytest.approx(6.0)
    assert _layout.smallest_type_pt(fig, placed_width_pt=252) == pytest.approx(3.0)


def test_shrink_resizes_and_lays_the_figure_out_again(new_figure):
    fig, ax = new_figure(figsize=(8, 5))
    ax.set_title("a title")
    ax.set_xlabel("x")
    fig.tight_layout()
    before = ax.get_position().bounds
    _layout.shrink(fig, 0.5)
    assert tuple(fig.get_size_inches()) == pytest.approx((4.0, 2.5))
    assert ax.get_position().bounds != pytest.approx(before)


def test_a_leader_line_through_another_label_is_found(new_figure, monkeypatch):
    """Placement avoids this, so it is forced: A's label goes to the bottom of
    the panel, and its line from A's point runs down through B's label."""
    import truecell._repel as repel

    fig, ax = new_figure(figsize=(4, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    a = ax.text(5, 8, "A", ha="center", va="center")
    b = ax.text(5, 5, "B", ha="center", va="center")
    repel.attach(ax, [a, b], [(5, 8), (5, 5)], [[(5, 8)], [(5, 5)]])

    def to_the_bottom(anchors, sizes, frame, groups, **kwargs):
        centres = np.array(anchors, dtype=float)
        centres[0, 1] = frame[1] + sizes[0][1]
        return centres, 0

    monkeypatch.setattr(repel, "place_labels", to_the_bottom)
    assert _layout.leaders_through_labels(fig) == [("A", "B")]
