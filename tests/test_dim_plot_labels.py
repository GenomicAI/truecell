"""Group labels in ``dim_plot``: where they sit, and what ``repel`` does.

Seurat's ``LabelClusters`` puts each label at the median of its group's
coordinates, per panel when the plot is split. ``repel`` answers
``DimPlot(repel = TRUE)``. Labels move apart, stay inside the panel, and leave
every group at least half visible. Every layout claim is checked with
``tests/_layout.py``, the check the tutorial figures pass, at the figure's own
size and at the 0.55 the manuscript prints tutorial figures at.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import truecell as tc
from truecell.dimreduc import DimReduc

plt = pytest.importorskip("matplotlib.pyplot")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _layout

from truecell._repel import RepelledLabels, _crosses, _leader_end, place_labels

NAMES = ["Naive CD4 T", "Memory CD4 T", "CD14+ Mono", "FCGR3A+ Mono", "CD8 T",
         "NK", "B", "Dendritic", "Platelet", "Plasmacytoid DC"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _object(embedding, labels, **columns):
    n = len(labels)
    cells = [f"C{i:04d}" for i in range(n)]
    obj = tc.create_truecell_object(
        counts=sp.csc_matrix(np.ones((3, n))), assay="RNA",
        feature_names=["G0", "G1", "G2"], cell_names=cells, project="labels")
    obj.reductions["umap"] = DimReduc(cell_embeddings=np.asarray(embedding, dtype=float),
                                      cell_names=cells, assay_used="RNA", key="umap_")
    obj.meta_data["ct"] = labels
    for name, values in columns.items():
        obj.meta_data[name] = values
    return obj


def crowded():
    """Ten groups a label's width apart in two rows: the labels collide."""
    rng = np.random.default_rng(1)
    xy, labels = [], []
    for i, name in enumerate(NAMES):
        xy.append(rng.normal((i % 5 * 1.0, (i // 5) * 0.6), 0.5, size=(80, 2)))
        labels += [name] * 80
    return np.vstack(xy), labels


def island():
    """Ten cells packed into a point inside a large group: smaller than their label."""
    rng = np.random.default_rng(2)
    big = rng.normal((0, 0), 1.0, size=(400, 2))
    tiny = rng.normal((1.0, 0.8), 0.01, size=(10, 2))
    return np.vstack([big, tiny]), ["Big cluster"] * 400 + ["Tiny"] * 10


def twins():
    """Two small groups at exactly the same place: either label hides both."""
    rng = np.random.default_rng(3)
    big = rng.normal((0, 0), 1.0, size=(400, 2))
    spot = np.tile((1.2, -0.9), (6, 1))
    return np.vstack([big, spot, spot]), ["Big cluster"] * 400 + ["Twin A"] * 6 + ["Twin B"] * 6


BUILDERS = {"crowded": crowded, "island": island, "twins": twins}


def _plot(obj, **kwargs):
    return tc.dim_plot(obj, group_by="ct", label=True, figsize=(7, 6), **kwargs)


def _positions(fig):
    return [(t.get_text(), tuple(t.get_position())) for ax in fig.axes for t in ax.texts]


# ---------------------------------------------------------------------------
# Where the labels belong
# ---------------------------------------------------------------------------

def test_labels_sit_at_the_median_as_label_clusters_puts_them():
    """A mean is pulled towards outlying cells; LabelClusters takes the median.
    With an even count the median is the middle two averaged, in R and numpy."""
    skewed = [(0, 0), (0, 1), (0, -1), (0, 0.5), (10, 0)]      # mean (2, 0.1)
    even = [(20, 0), (21, 0), (22, 1), (30, 1)]                 # median (21.5, 0.5)
    obj = _object(skewed + even, ["skewed"] * 5 + ["even"] * 4)
    positions = dict(_positions(_plot(obj)))
    assert positions["skewed"] == (0.0, 0.0)
    assert positions["even"] == (21.5, 0.5)


def test_a_split_plot_labels_each_panel_at_that_panels_median():
    xy = [(0, 0), (1, 0), (2, 0), (10, 5), (11, 5), (12, 5)]
    obj = _object(xy, ["g"] * 6, stim=["CTRL"] * 3 + ["STIM"] * 3)
    fig = _plot(obj, split_by="stim")
    by_panel = {ax.get_title(): [tuple(t.get_position()) for t in ax.texts]
                for ax in fig.axes if ax.get_visible()}
    assert by_panel == {"CTRL": [(1.0, 0.0)], "STIM": [(11.0, 5.0)]}


# ---------------------------------------------------------------------------
# What goes wrong without repel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build,expected", [
    ("crowded", "text overlaps text"),
    ("island", "label 'Tiny' hides 10 of 10 markers of 'Tiny'"),
    ("twins", "label 'Twin A' hides 6 of 6 markers of 'Twin B'"),
])
def test_without_repel_the_fixtures_show_the_defect(build, expected):
    """Premise for the repel tests below: each fixture breaks the layout check."""
    emb, labels = BUILDERS[build]()
    problems = _layout.check_layout(_plot(_object(emb, labels)), on_data=set(labels))
    assert any(p.startswith(expected) for p in problems), problems


# ---------------------------------------------------------------------------
# repel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", sorted(BUILDERS))
def test_repel_leaves_no_label_overlapping_or_any_group_hidden(build):
    emb, labels = BUILDERS[build]()
    fig = _plot(_object(emb, labels), repel=True)
    assert _layout.check_layout(fig, on_data=set(labels)) == []


@pytest.mark.parametrize("build", sorted(BUILDERS))
def test_repel_still_holds_when_the_figure_is_printed_smaller(build):
    """Placement is worked out at draw time. Fixed when the plot was built, it
    would collide again once the manuscript shrinks the figure around its text."""
    emb, labels = BUILDERS[build]()
    fig = _plot(_object(emb, labels), repel=True)
    fig.canvas.draw()
    _layout.shrink(fig)
    assert _layout.check_layout(fig, on_data=set(labels)) == []


def test_repel_moves_no_label_that_needs_no_move():
    rng = np.random.default_rng(5)
    centres = [(0, 0), (10, 0), (5, 9)]
    xy = np.vstack([rng.normal(c, 1.5, size=(300, 2)) for c in centres])
    labels = [name for name in ("left", "right", "top") for _ in range(300)]
    obj = _object(xy, labels)
    still = _plot(obj)
    moved = _plot(obj, repel=True)
    moved.canvas.draw()
    assert _positions(moved) == _positions(still)
    assert not any(line.get_visible() for line in moved.axes[0].lines)


def _box_px(text, renderer):
    patch = text.get_bbox_patch()
    pad = patch.get_boxstyle().pad * renderer.points_to_pixels(text.get_fontsize())
    return text.get_window_extent(renderer).padded(pad)


@pytest.mark.parametrize("build,shrink", [("crowded", True), ("twins", True)])
def test_a_label_moved_off_its_group_is_joined_to_it_by_a_line(build, shrink):
    """A line runs from the group's median to the nearest edge of its label
    exactly when the label has moved more than half its height away."""
    emb, labels = BUILDERS[build]()
    obj = _object(emb, labels)
    fig = _plot(obj, repel=True)
    if shrink:
        _layout.shrink(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    artist = next(a for a in ax.get_children() if isinstance(a, RepelledLabels))
    groups = np.asarray(labels)
    joined = 0
    for text, line in zip(artist._texts, artist._leaders):
        median = np.median(emb[groups == text.get_text()], axis=0)
        anchor = ax.transData.transform(median)
        box = _box_px(text, renderer)
        nearest = np.clip(anchor, (box.x0, box.y0), (box.x1, box.y1))
        distance = np.hypot(*(anchor - nearest))
        if abs(distance - box.height / 2) < 1.0:
            continue                                   # too close to call by a pixel
        assert line.get_visible() == (distance > box.height / 2), text.get_text()
        if line.get_visible():
            joined += 1
            start, end = ax.transData.transform(line.get_xydata())
            assert np.allclose(start, anchor, atol=0.01)
            assert np.allclose(end, nearest, atol=1.0)
    assert joined, "premise: this fixture moves at least one label that far"


def test_repel_places_the_labels_the_same_way_every_time():
    emb, labels = crowded()
    first, second = (_plot(_object(emb, labels), repel=True) for _ in range(2))
    for fig in (first, second):
        fig.canvas.draw()
    assert _positions(first) == _positions(second)
    assert _positions(first) != _positions(_plot(_object(emb, labels)))


def test_a_second_draw_changes_nothing(monkeypatch):
    """Every change marks the figure stale, and an interactive backend answers a
    stale figure by drawing it again, so a draw that re-set the same positions
    would never settle."""
    emb, labels = crowded()
    fig = _plot(_object(emb, labels), repel=True)
    fig.canvas.draw()
    calls = []
    ax = fig.axes[0]
    for artist in list(ax.texts) + list(ax.lines):
        for method in ("set_position", "set_data", "set_visible"):
            if hasattr(artist, method):
                original = getattr(artist, method)
                monkeypatch.setattr(artist, method,
                                    lambda *a, _o=original, _m=method, **k: (calls.append(_m), _o(*a, **k)))
    fig.canvas.draw()
    assert calls == []


def test_labels_that_can_not_all_fit_warn_once():
    rng = np.random.default_rng(4)
    names = [f"a long cluster name {i}" for i in range(12)]
    obj = _object(rng.normal(0, 1, size=(240, 2)), [names[i % 12] for i in range(240)])
    fig = tc.dim_plot(obj, group_by="ct", label=True, repel=True, figsize=(3, 3))
    with pytest.warns(UserWarning, match="could not be kept apart"):
        fig.canvas.draw()
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        fig.canvas.draw()
    assert not [w for w in again if "could not be kept apart" in str(w.message)]


def test_a_split_plot_repels_in_every_panel():
    emb, labels = crowded()
    stim = ["CTRL" if i % 2 else "STIM" for i in range(len(labels))]
    fig = tc.dim_plot(_object(emb, labels, stim=stim), group_by="ct", split_by="stim",
                      label=True, repel=True)
    assert sum(isinstance(a, RepelledLabels) for ax in fig.axes for a in ax.get_children()) == 2
    assert _layout.check_layout(fig, on_data=set(labels), tight_bbox=True) == []


def test_a_figure_with_repelled_labels_survives_pickling():
    emb, labels = crowded()
    fig = _plot(_object(emb, labels), repel=True)
    copy = pickle.loads(pickle.dumps(fig))
    assert _layout.check_layout(copy, on_data=set(labels)) == []
    plt.close(copy)


# ---------------------------------------------------------------------------
# The placement on its own
# ---------------------------------------------------------------------------

FRAME = (0.0, 0.0, 400.0, 300.0)


def _spread(centre, n=200, sd=60.0, seed=0):
    return np.random.default_rng(seed).normal(centre, sd, size=(n, 2))


def test_two_labels_on_one_point_both_leave_it():
    """Neither label may cover the other's anchor, so neither stays on the point.
    Square boxes, so moving up and moving right clear at the same distance and
    only the documented order decides where the first goes: nearest first, then
    clockwise from up. The larger group is placed first."""
    anchors = [(200, 150), (200, 150)]
    sizes = [(20, 20), (20, 20)]
    groups = [_spread((200, 150), n=200, seed=1), _spread((200, 150), n=300)]
    centres, unplaced = place_labels(anchors, sizes, FRAME, groups, gap=2.0)
    assert unplaced == 0
    first, second = centres[1], centres[0]
    assert first[0] == 200.0 and first[1] - 10 > 150 + 2       # straight up, clear of the point
    for x, y in (first, second):
        assert abs(x - 200) > 10 + 2 or abs(y - 150) > 10 + 2  # neither covers the point
    assert abs(first[0] - second[0]) >= 22 or abs(first[1] - second[1]) >= 22


def _crossings(anchors, sizes, centres):
    """``(i, j)`` where label i's leader line runs through label j's box, found by
    sampling along the line rather than with the placement's own geometry."""
    found = []
    for i, (anchor, centre, size) in enumerate(zip(anchors, centres, sizes)):
        end = _leader_end(np.asarray(anchor, float), centre, np.asarray(size, float))
        if end is None:
            continue
        t = np.linspace(0.0, 1.0, 400)[:, None]
        points = np.asarray(anchor, float) + t * (end - np.asarray(anchor, float))
        for j, (other, other_size) in enumerate(zip(centres, sizes)):
            half = np.asarray(other_size, float) / 2 - 1
            if j != i and np.any(np.all(np.abs(points - other) < half, axis=1)):
                found.append((i, j))
    return found


def test_a_leader_line_goes_round_labels_when_it_can(monkeypatch):
    """A small label whose point is ringed by five others. Nearest-first, its
    line runs through one of them; a position further out leaves a clear line."""
    import truecell._repel as repel

    anchors = [(179, 109), (157, 178), (237, 160), (221, 154), (239, 178), (200, 150)]
    sizes = [(60, 20), (103, 20), (62, 20), (97, 20), (69, 20), (83, 20)]
    rng = np.random.default_rng(0)
    groups = [rng.normal(a, 80, size=(300, 2)) for a in anchors[:-1]] + [np.array([anchors[-1]], float)]

    centres, _ = place_labels(anchors, sizes, FRAME, groups, gap=2.0)
    assert _crossings(anchors, sizes, centres) == []

    monkeypatch.setattr(repel, "_crossing", lambda anchor, c, *args: np.zeros(len(c), dtype=bool))
    careless, _ = place_labels(anchors, sizes, FRAME, groups, gap=2.0)
    assert _crossings(anchors, sizes, careless), "premise: nearest-first draws a crossing"


def _hidden(anchors, sizes, centres, groups, gap=2.0):
    """``(label, what)`` where a label covers more than half of a group's cells or
    another label's anchor, counting up to *gap* outside its box as covered."""
    found = []
    for j, (centre, size) in enumerate(zip(centres, sizes)):
        half = np.asarray(size, float) / 2 + gap
        for k, cells in enumerate(groups):
            under = np.all(np.abs(np.asarray(cells, float) - centre) < half, axis=1).sum()
            if under > np.floor(0.5 * len(cells)):
                found.append((j, f"group {k}"))
        for k, anchor in enumerate(anchors):
            if k != j and np.all(np.abs(np.asarray(anchor, float) - centre) < half):
                found.append((j, f"anchor {k}"))
    return found


def _groups(anchors, small):
    """Tiny groups stacked on their anchor; the rest spread wide enough never to hide."""
    return [np.tile(np.asarray(a, float), (8, 1)) if k in small
            else np.random.default_rng(k).normal(a, 60.0, size=(300, 2))
            for k, a in enumerate(anchors)]


# Layouts found by searching random ones for cases that nearest-first placement
# gets wrong, each decided by a different part of place_labels.
EIGHT = ((0.0, 0.0, 200.0, 150.0),
         [(118.5, 117.6), (89.8, 70.2), (121.5, 66.2), (102.8, 120.3),
          (58.8, 69.0), (150.4, 111.5), (51.7, 69.3), (61.9, 42.4)],
         [(109.8, 20.0), (46.2, 20.0), (32.9, 20.0), (90.2, 20.0),
          (55.8, 20.0), (31.9, 20.0), (72.7, 20.0), (104.8, 20.0)], {1, 4})
FIVE = ((0.0, 0.0, 200.0, 220.0),
        [(96.1, 122.3), (103.8, 130.1), (121.7, 117.6), (122.4, 102.2), (73.1, 107.0)],
        [(44.8, 20.0), (93.4, 20.0), (84.3, 20.0), (103.7, 20.0), (70.9, 20.0)], {0})


def test_eight_crowded_labels_leave_no_line_through_a_label():
    """Needs a label kept off a leader line already drawn, and a search of the
    whole frame when nothing within reach of the anchor is clear."""
    frame, anchors, sizes, small = EIGHT
    groups = _groups(anchors, small)
    centres, unplaced = place_labels(anchors, sizes, frame, groups, gap=2.0)
    assert unplaced == 0
    assert _crossings(anchors, sizes, centres) == []
    assert _hidden(anchors, sizes, centres, groups) == []


def test_a_label_left_no_clear_line_still_hides_nothing():
    """Here one label has no position whose line misses every label. It takes one
    that hides nothing rather than the nearest, which sits on a group."""
    frame, anchors, sizes, small = FIVE
    groups = _groups(anchors, small)
    centres, unplaced = place_labels(anchors, sizes, frame, groups, gap=2.0)
    assert unplaced == 0
    assert _crossings(anchors, sizes, centres), "premise: a crossing is left"
    assert _hidden(anchors, sizes, centres, groups) == []


@pytest.mark.parametrize("start,end,expected", [
    ((0, 5), (10, 5), False),      # stops short of the box
    ((0, 5), (22, 5), True),       # runs into it
    ((25, 5), (40, 5), True),      # starts inside it
    ((0, 0), (40, 0), False),      # level with it, below
    ((25, -10), (25, 20), True),   # upright, straight through
    ((0, 20), (40, 12), False),    # slopes past above it
])
def test_a_segment_crosses_a_box_only_where_it_runs_through_it(start, end, expected):
    box = np.array([[20.0, 1.0, 30.0, 10.0]])
    result = _crosses(np.array([start], dtype=float), np.array([end], dtype=float), box)
    assert result.shape == (1, 1)
    assert bool(result[0, 0]) is expected


def test_twins_leader_line_misses_the_other_labels():
    emb, labels = twins()
    fig = _layout.shrink(_plot(_object(emb, labels), repel=True))
    fig.canvas.draw()
    assert any(line.get_visible() for line in fig.axes[0].lines), "premise: a leader is drawn"
    assert _layout.leaders_through_labels(fig) == []


def test_a_label_by_the_edge_is_pulled_inside_the_frame():
    centres, _ = place_labels([(10, 150)], [(80, 20)], FRAME, [_spread((10, 150))], gap=2.0)
    assert centres[0][0] - 40 >= 2.0


def test_a_group_smaller_than_its_label_is_left_mostly_in_view():
    cells = np.random.default_rng(2).normal((200, 150), 2.0, size=(10, 2))
    centres, _ = place_labels([(200, 150)], [(80, 20)], FRAME, [cells], gap=2.0)
    x, y = centres[0]
    covered = ((np.abs(cells[:, 0] - x) < 40) & (np.abs(cells[:, 1] - y) < 10)).sum()
    assert covered <= 5


def test_labels_with_no_room_near_their_anchor_go_elsewhere_in_the_frame():
    """Forty labels on one point: the nearest free places run out within reach
    of the anchor, and the rest of the frame still has room for all of them."""
    frame = (0.0, 0.0, 1000.0, 400.0)
    n = 40
    centres, unplaced = place_labels([(100, 200)] * n, [(80, 20)] * n, frame,
                                     [_spread((100, 200), sd=300, seed=s) for s in range(n)],
                                     gap=2.0)
    assert unplaced == 0
    far = np.hypot(*(centres - (100, 200)).T) > 2 * (80 + 20)
    assert far.any(), "premise: some labels had to leave the anchor's reach"
    for i in range(n):
        for j in range(i + 1, n):
            apart = (abs(centres[i][0] - centres[j][0]) >= 80 + 2
                     or abs(centres[i][1] - centres[j][1]) >= 20 + 2)
            assert apart, (i, j)


def test_labels_with_no_room_anywhere_are_counted():
    frame = (0.0, 0.0, 100.0, 30.0)
    anchors = [(50, 15)] * 3
    centres, unplaced = place_labels(anchors, [(90, 20)] * 3, frame,
                                     [_spread((50, 15), sd=30, seed=s) for s in range(3)], gap=2.0)
    assert unplaced == 2
    assert np.all(np.isfinite(centres))


@pytest.mark.parametrize("anchor,expected", [
    ((100, 100), None),              # under the label
    ((100, 115), None),              # 5 px above the top edge, under half the height
    ((100, 140), (100, 110)),        # 30 px above: joined to the top edge
    ((200, 60), (140, 90)),          # off a corner: joined to the corner
])
def test_leader_lines_run_to_the_nearest_edge_only_when_needed(anchor, expected):
    end = _leader_end(np.array(anchor, float), np.array((100.0, 100.0)), np.array((80.0, 20.0)))
    if expected is None:
        assert end is None
    else:
        assert tuple(end) == expected
