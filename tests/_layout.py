"""Layout checks for a rendered matplotlib figure.

Upstreamed from the Frontiers revision's figure build (``build/figstyle.py``),
where every manuscript figure has to pass them before it is written. Reviewer 3
asked for a systematic answer to overlapping annotations rather than panels
nudged by hand, and a check that fails is what makes one systematic.

:func:`check_layout` draws the figure and lists what a reader would see as
broken:

* text overlapping other text;
* text off the canvas, unless the figure is saved with ``bbox_inches="tight"``,
  which grows the canvas to fit it;
* a legend over plotted markers or bars;
* an annotation over a legend, a bar, or plotted markers.

Group labels drawn on the data by design, such as ``dim_plot``'s cluster names,
are passed as ``on_data``. They may sit on markers, but must not hide more than
half of any group's markers: that is how a label box makes a small cluster
disappear.

Changes from the paper's version, each found by running it over the tutorial
figures:

* tick labels for ticks outside the view interval are skipped: matplotlib keeps
  them visible without drawing them, so they read as text off the canvas;
* rotated text is compared as the rotated rectangle it is, since the upright
  boxes of 45-degree labels overlap where the text does not;
* bars and Visium's spot ellipses count as data, because the integration
  scoreboard's legend and value labels sat on its bars;
* labels on the data are checked for hiding a group, where the paper exempted
  them outright.

:func:`shrink` lays a figure out again at a fraction of its size, the way the
manuscript places tutorial figures. Type keeps its point size while the axes
shrink, so collisions show up there first.
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.collections import EllipseCollection, PathCollection
from matplotlib.container import BarContainer
from matplotlib.text import Text
from matplotlib.transforms import Bbox

# A label may cover at most this share of a group's markers.
HIDE_FRACTION = 0.5

# Overlaps smaller than this many pixels are rounding, not collisions.
_TOL_PX = 1.0

# The manuscript prints tutorial figures at 0.55 of their authored size.
MANUSCRIPT_SCALE = 0.55


def _renderer(fig):
    fig.canvas.draw()
    get = getattr(fig.canvas, "get_renderer", None)
    return get() if get is not None else fig._get_renderer()


def _tick_labels(axis) -> list[Text]:
    """The tick labels matplotlib draws.

    Ticks outside the view interval keep visible labels that are never drawn.
    Listing them all reported a UMAP's ``−10`` as text off the canvas.
    """
    lo, hi = sorted(axis.get_view_interval())
    eps = 1e-9 * max(1.0, abs(hi - lo))
    labels: list[Text] = []
    for locs, ticks in ((axis.get_majorticklocs(), axis.get_major_ticks),
                        (axis.get_minorticklocs(), axis.get_minor_ticks)):
        for loc, tick in zip(locs, ticks(len(locs))):
            if lo - eps <= loc <= hi + eps:
                labels += [tick.label1, tick.label2]
    return labels


def texts(fig) -> list[Text]:
    """Every visible, non-empty piece of text on the figure."""
    out: list[Text] = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        out += list(ax.texts)
        out += [ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)]
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                if axis.get_visible():
                    out.append(axis.label)
                    out += _tick_labels(axis)
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            out += list(legend.get_texts()) + [legend.get_title()]
    for legend in fig.legends:
        if legend.get_visible():
            out += list(legend.get_texts()) + [legend.get_title()]
    return [t for t in out
            if t is not None and t.get_visible() and t.get_text().strip()]


def _pad(text: Text, renderer) -> float:
    """How far the box drawn behind *text*, if any, reaches past it, in pixels."""
    patch = text.get_bbox_patch()
    if patch is None or not patch.get_visible():
        return 0.0
    return getattr(patch.get_boxstyle(), "pad", 0.0) * renderer.points_to_pixels(text.get_fontsize())


def _box(text: Text, renderer) -> Bbox:
    """The text's upright extent, including the box drawn behind it if there is one."""
    return text.get_window_extent(renderer).padded(_pad(text, renderer))


def _corners(text: Text, renderer) -> np.ndarray:
    """The corners of the text's box in pixels, turned with the text.

    A rotated label's window extent is the upright box around it. Two labels at
    45 degrees, as under a violin plot, have upright boxes that overlap where
    their text does not, so rotated text is compared as the rectangle it is.
    """
    upright = _box(text, renderer)
    rotation = text.get_rotation()
    if rotation % 90 == 0:
        return np.array([[upright.x0, upright.y0], [upright.x1, upright.y0],
                         [upright.x1, upright.y1], [upright.x0, upright.y1]])
    text.set_rotation(0)
    try:
        flat = _box(text, renderer)
    finally:
        text.set_rotation(rotation)
    # A rectangle's upright bounding box is centred on the rectangle.
    half = np.array([flat.width, flat.height]) / 2
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    c, s = np.cos(np.deg2rad(rotation)), np.sin(np.deg2rad(rotation))
    return local @ np.array([[c, s], [-s, c]]) + (
        (upright.x0 + upright.x1) / 2, (upright.y0 + upright.y1) / 2)


def _overlap(a: Bbox, b: Bbox, tol: float = _TOL_PX) -> bool:
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol
            and min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def _rectangles_overlap(p: np.ndarray, q: np.ndarray, tol: float = _TOL_PX) -> bool:
    """Whether two rectangles given by their corners overlap by more than *tol*.

    Two convex shapes are apart if some edge direction separates them; for
    rectangles, checking the edges of both is enough.
    """
    for shape in (p, q):
        for k in range(4):
            edge = shape[(k + 1) % 4] - shape[k]
            length = np.hypot(*edge)
            if length == 0:
                continue
            axis = np.array([-edge[1], edge[0]]) / length
            a, b = p @ axis, q @ axis
            if min(a.max(), b.max()) - max(a.min(), b.min()) <= tol:
                return False
    return True


def _inside(points: np.ndarray, box: Bbox, pad: float = 0.0) -> int:
    if not len(points):
        return 0
    return int(np.count_nonzero(
        (points[:, 0] > box.x0 - pad) & (points[:, 0] < box.x1 + pad)
        & (points[:, 1] > box.y0 - pad) & (points[:, 1] < box.y1 + pad)))


def _marker_sets(ax, dpi: float) -> list[tuple[str, np.ndarray, float]]:
    """``(label, points in pixels, marker radius in pixels)`` per marker layer.

    Scatter markers, and the spots Visium plots draw as ellipses, whose radius
    is taken as zero. Points outside the axes are clipped away when drawn, so
    they are dropped: a legend beside the axes can not cover them.
    """
    frame = ax.bbox
    sets = []

    def in_view(points):
        keep = ((points[:, 0] >= frame.x0) & (points[:, 0] <= frame.x1)
                & (points[:, 1] >= frame.y0) & (points[:, 1] <= frame.y1))
        return points[keep]

    for coll in ax.collections:
        if not isinstance(coll, (PathCollection, EllipseCollection)) or not coll.get_visible():
            continue
        offsets = np.asarray(coll.get_offsets(), dtype=float)
        if not len(offsets):
            continue
        sizes = coll.get_sizes() if isinstance(coll, PathCollection) else []
        radius = np.sqrt(sizes.max()) * dpi / 72.0 / 2.0 if len(sizes) else 0.0
        points = coll.get_offset_transform().transform(offsets)
        sets.append((coll.get_label(), in_view(points) if coll.get_clip_on() else points, radius))
    for line in ax.get_lines():
        if not line.get_visible() or line.get_marker() in (None, "None", "", " "):
            continue
        xy = np.asarray(line.get_xydata(), dtype=float)
        if not len(xy):
            continue
        points = ax.transData.transform(xy)
        radius = line.get_markersize() * dpi / 72.0 / 2.0
        sets.append((line.get_label(), in_view(points) if line.get_clip_on() else points, radius))
    return sets


def _bar_boxes(ax, renderer) -> list[Bbox]:
    boxes = []
    for container in ax.containers:
        if not isinstance(container, BarContainer):
            continue
        for bar in container.patches:
            if not bar.get_visible() or bar.get_width() * bar.get_height() == 0:
                continue
            box = Bbox.intersection(bar.get_window_extent(renderer), ax.bbox)
            if box is not None:
                boxes.append(box)
    return boxes


def _panel(ax) -> str:
    title = ax.get_title()
    return f"panel {title!r}" if title else "an untitled panel"


def check_layout(fig, *, allow=None, on_data=(), tight_bbox: bool = False) -> list[str]:
    """Draw *fig* and return its layout problems, sorted; empty when it is clean.

    Parameters
    ----------
    allow      : pairs of strings that may overlap each other.
    on_data    : strings of labels drawn on the data on purpose. They may cover
                 markers, but not more than :data:`HIDE_FRACTION` of any one
                 marker layer (in ``dim_plot``, one layer is one group).
    tight_bbox : the figure is saved with ``bbox_inches="tight"``, which grows
                 the canvas to fit its text, so text past the edge is kept.
    """
    allow = {tuple(pair) for pair in (allow or ())}
    on_data = set(on_data)
    renderer = _renderer(fig)
    problems: list[str] = []

    shapes = [(t, _corners(t, renderer)) for t in texts(fig)]
    if not tight_bbox:
        canvas = fig.bbox
        for t, corners in shapes:
            if ((corners < (canvas.x0 - _TOL_PX, canvas.y0 - _TOL_PX)).any()
                    or (corners > (canvas.x1 + _TOL_PX, canvas.y1 + _TOL_PX)).any()):
                problems.append(f"text off the canvas: {t.get_text()!r}")
    if shapes:
        # Upright boxes first, vectorised, so only pairs that might touch get the
        # exact test: a heatmap's gene labels alone make tens of thousands of pairs.
        low = np.array([corners.min(axis=0) for _, corners in shapes])
        high = np.array([corners.max(axis=0) for _, corners in shapes])
        near = np.triu(
            (np.minimum(high[:, None, 0], high[None, :, 0])
             - np.maximum(low[:, None, 0], low[None, :, 0]) > _TOL_PX)
            & (np.minimum(high[:, None, 1], high[None, :, 1])
               - np.maximum(low[:, None, 1], low[None, :, 1]) > _TOL_PX), k=1)
        for i, j in zip(*np.nonzero(near)):
            (t1, c1), (t2, c2) = shapes[i], shapes[j]
            pair = (t1.get_text(), t2.get_text())
            if pair in allow or pair[::-1] in allow:
                continue
            if _rectangles_overlap(c1, c2):
                problems.append(f"text overlaps text: {pair[0]!r} / {pair[1]!r}")

    axes = [ax for ax in fig.axes if ax.get_visible()]
    figure_legends = [leg for leg in fig.legends if leg.get_visible()]
    legends = figure_legends + [ax.get_legend() for ax in axes
                                if ax.get_legend() is not None and ax.get_legend().get_visible()]

    for ax in axes:
        markers = _marker_sets(ax, fig.dpi)
        bars = _bar_boxes(ax, renderer)
        own = ax.get_legend()
        covering = figure_legends + ([own] if own is not None and own.get_visible() else [])
        for legend in covering:
            box = legend.get_window_extent(renderer)
            covered = sum(_inside(points, box) for _, points, _ in markers)
            if covered:
                problems.append(f"legend covers {covered} marker(s) in {_panel(ax)}")
            n_bars = sum(_overlap(box, bar) for bar in bars)
            if n_bars:
                problems.append(f"legend covers {n_bars} bar(s) in {_panel(ax)}")

        for t in ax.texts:
            if not (t.get_visible() and t.get_text().strip()):
                continue
            text, box = t.get_text(), _box(t, renderer)
            if any(_overlap(box, legend.get_window_extent(renderer)) for legend in legends):
                problems.append(f"annotation overlaps a legend: {text!r}")
            n_bars = sum(_overlap(box, bar) for bar in bars)
            if n_bars:
                problems.append(f"annotation overlaps {n_bars} bar(s): {text!r}")
            if text in on_data:
                for layer, points, _ in markers:
                    hidden = _inside(points, box)
                    if hidden > HIDE_FRACTION * len(points):
                        problems.append(f"label {text!r} hides {hidden} of {len(points)} "
                                        f"markers of {layer!r}")
                continue
            covered = sum(_inside(points, box, pad=radius) for _, points, radius in markers)
            if covered:
                problems.append(f"annotation overlaps {covered} marker(s): {text!r}")
    return sorted(set(problems))


def smallest_type_pt(fig, placed_width_pt: float | None = None) -> float:
    """The smallest type on *fig*, in points as printed.

    ``placed_width_pt`` is the width the figure is printed at, such as a journal's
    504 pt text block. Matplotlib sizes type in points of the figure's own width,
    so a 9-inch figure placed at 504 pt prints 10 pt labels at 7.8 pt.
    """
    _renderer(fig)
    scale = 1.0 if placed_width_pt is None else placed_width_pt / (fig.get_figwidth() * 72.0)
    sizes = [t.get_fontsize() * scale for t in texts(fig)]
    return min(sizes) if sizes else float("inf")


def shrink(fig, scale: float = MANUSCRIPT_SCALE):
    """Resize *fig* to *scale* of its size and lay it out again; returns *fig*.

    This is what ``regen_tutorial_figures.py`` does before it saves a tutorial
    figure for the manuscript: every string keeps its point size while the axes
    shrink around it. The resize is not undone, so check the full-size figure
    first.
    """
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    width, height = fig.get_size_inches()
    fig.set_size_inches(width * scale, height * scale)
    if not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    return fig


def leaders_through_labels(fig) -> list[tuple[str, str]]:
    """``(label, other label)`` wherever a repelled label's leader line runs
    through another label's box. ``check_layout`` does not see lines at all.

    Checked by sampling along the drawn segment, not with the geometry the
    placement itself uses, so the two can not share a mistake.
    """
    from truecell._repel import RepelledLabels

    renderer = _renderer(fig)
    crossings = []
    for ax in fig.axes:
        for artist in ax.get_children():
            if not isinstance(artist, RepelledLabels):
                continue
            boxes = [(text, _box(text, renderer)) for text in artist._texts]
            for own, line in zip(artist._texts, artist._leaders):
                if not line.get_visible():
                    continue
                (x0, y0), (x1, y1) = ax.transData.transform(np.asarray(line.get_xydata(), dtype=float))
                t = np.linspace(0.0, 1.0, 400)
                xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                for other, box in boxes:
                    if other is own:
                        continue
                    if np.any((xs > box.x0 + _TOL_PX) & (xs < box.x1 - _TOL_PX)
                              & (ys > box.y0 + _TOL_PX) & (ys < box.y1 - _TOL_PX)):
                        crossings.append((own.get_text(), other.get_text()))
    return crossings
