"""Group labels that keep clear of each other, placed each time they are drawn.

``dim_plot(label=True, repel=True)`` answers Seurat's ``DimPlot(label = TRUE,
repel = TRUE)``, where ggrepel pushes the cluster labels apart. ggrepel is GPL-3,
so this is not a port of it. It is a greedy placement (see :func:`place_labels`),
and it is deterministic, where ggrepel starts from random jitter.

Placement happens at draw time. How much room a label needs depends on its size
relative to the axes, and that changes whenever the figure is resized, laid out
again or saved at another size. The manuscript prints tutorial figures at 0.55 of
their size. Positions fixed when the plot was built would collide again there, so
:class:`RepelledLabels` moves the labels on every draw.

This module imports matplotlib at the top, and matplotlib is an optional
dependency, so ``plotting`` imports it only when a plot asks to repel.
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D

# A label may cover at most this share of any group's cells. truecell draws its
# labels in a white box, and at a group's median that box hid any group smaller
# than itself: all 14 cells of the pbmc3k platelet island sat under "Platelet".
HIDE_FRACTION = 0.5

# Space kept between two labels, and between a label and the panel edge.
GAP_PT = 2.0

# Drawn after the cells (collections draw at zorder 1) and before the leader lines
# and the labels (text draws at 3), so both have moved by the time they draw.
ZORDER = 1.5
LEADER_ZORDER = 2.5

# Above this many cells, a group's share under a label is counted on an evenly
# spaced subsample, which bounds the cost of a draw.
_MAX_CELLS = 5000
_CHUNK = 256


def _offsets(reach: float, step: float) -> np.ndarray:
    """Lattice offsets within *reach*: nearest first, then clockwise from up."""
    k = int(reach // step)
    steps = np.arange(-k, k + 1)
    dx, dy = (a.ravel() for a in np.meshgrid(steps, steps))
    d2 = dx * dx + dy * dy
    keep = d2 <= k * k
    dx, dy, d2 = dx[keep], dy[keep], d2[keep]
    order = np.lexsort((np.mod(np.arctan2(dx, dy), 2 * np.pi), d2))
    return np.column_stack([dx[order], dy[order]]) * step


def _in_frame(centres: np.ndarray, w: float, h: float, frame, gap: float) -> np.ndarray:
    x0, y0, x1, y1 = frame
    return ((centres[:, 0] - w / 2 >= x0 + gap) & (centres[:, 0] + w / 2 <= x1 - gap)
            & (centres[:, 1] - h / 2 >= y0 + gap) & (centres[:, 1] + h / 2 <= y1 - gap))


def _clear(centres: np.ndarray, w: float, h: float, placed: list, gap: float) -> np.ndarray:
    """Which centres keep *gap* from every box already placed."""
    if not placed:
        return np.ones(len(centres), dtype=bool)
    boxes = np.asarray(placed, dtype=float)  # x, y, width, height
    dx = np.abs(centres[:, None, 0] - boxes[None, :, 0])
    dy = np.abs(centres[:, None, 1] - boxes[None, :, 1])
    hit = (dx < (w + boxes[None, :, 2]) / 2 + gap) & (dy < (h + boxes[None, :, 3]) / 2 + gap)
    return ~hit.any(axis=1)


def _narrowest(values: np.ndarray, count: int) -> float:
    """Width of the narrowest interval that holds *count* of *values*."""
    v = np.sort(values)
    return float(np.min(v[count - 1:] - v[:len(v) - count + 1]))


def _groups_at_risk(groups, hide: float) -> list:
    """``(cells, allowed, narrowest width, narrowest height)`` per group.

    A box can cover more than *allowed* of a group's cells only if it is wider and
    taller than the narrowest intervals holding one cell more than that, so a
    group whose intervals are larger than a label never needs counting.
    """
    table = []
    for cells in groups:
        cells = np.asarray(cells, dtype=float).reshape(-1, 2)
        cells = cells[np.isfinite(cells).all(axis=1)]
        if len(cells) > _MAX_CELLS:
            cells = cells[::-(-len(cells) // _MAX_CELLS)]
        allowed = int(np.floor(hide * len(cells)))
        if allowed + 1 > len(cells):
            continue
        table.append((cells, allowed, _narrowest(cells[:, 0], allowed + 1),
                      _narrowest(cells[:, 1], allowed + 1)))
    return table


def _hiding(centres: np.ndarray, w: float, h: float, risky: list) -> np.ndarray:
    """Which centres put a box over more than the allowed cells of some group."""
    hides = np.zeros(len(centres), dtype=bool)
    x0, x1 = centres[:, :1] - w / 2, centres[:, :1] + w / 2
    y0, y1 = centres[:, 1:] - h / 2, centres[:, 1:] + h / 2
    for cells, allowed in risky:
        under = ((cells[:, 0] > x0) & (cells[:, 0] < x1)
                 & (cells[:, 1] > y0) & (cells[:, 1] < y1)).sum(axis=1)
        hides |= under > allowed
    return hides


def _crosses(starts: np.ndarray, ends: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """``(segments, boxes)``: whether each segment passes through each box's inside.

    Clips each segment to each box one axis at a time: it crosses the box when the
    stretch of it inside the box's x range overlaps the stretch inside its y range.
    Boxes are ``(x0, y0, x1, y1)``.
    """
    if not len(starts) or not len(boxes):
        return np.zeros((len(starts), len(boxes)), dtype=bool)
    p = starts[:, None, :]
    d = (ends - starts)[:, None, :]
    lo, hi = boxes[None, :, :2], boxes[None, :, 2:]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = (lo - p) / d, (hi - p) / d
    near, far = np.minimum(t0, t1), np.maximum(t0, t1)
    # A segment parallel to an axis is inside that axis's range throughout, or never.
    flat = np.broadcast_to(d == 0, near.shape)
    inside = (p > lo) & (p < hi)
    near = np.where(flat, np.where(inside, -np.inf, np.inf), near)
    far = np.where(flat, np.where(inside, np.inf, -np.inf), far)
    return np.maximum(near.max(axis=2), 0.0) < np.minimum(far.min(axis=2), 1.0)


def _leader_ends(anchor: np.ndarray, centres: np.ndarray, w: float, h: float):
    """Where each candidate's leader line would meet its box, and which need one."""
    half = np.array([w, h]) / 2
    ends = centres + np.clip(anchor - centres, -half, half)
    return ends, np.hypot(*(anchor - ends).T) > h / 2


def _crossing(anchor, centres, w, h, placed: list, leaders: list) -> np.ndarray:
    """Which centres put a box across a leader line already drawn, or would draw
    a leader line across a label already placed."""
    boxes = np.column_stack([centres[:, 0] - w / 2, centres[:, 1] - h / 2,
                             centres[:, 0] + w / 2, centres[:, 1] + h / 2])
    crossing = np.zeros(len(centres), dtype=bool)
    if leaders:
        lines = np.asarray(leaders, dtype=float)
        crossing |= _crosses(lines[:, :2], lines[:, 2:], boxes).any(axis=0)
    if placed:
        others = np.asarray(placed, dtype=float)
        other_boxes = np.column_stack([others[:, 0] - others[:, 2] / 2, others[:, 1] - others[:, 3] / 2,
                                       others[:, 0] + others[:, 2] / 2, others[:, 1] + others[:, 3] / 2])
        ends, needed = _leader_ends(anchor, centres, w, h)
        if needed.any():
            starts = np.broadcast_to(anchor, (int(needed.sum()), 2))
            crossing[needed] |= _crosses(starts, ends[needed], other_boxes).any(axis=1)
    return crossing


def _free_grid(anchor: np.ndarray, w: float, h: float, frame, placed: list, gap: float) -> np.ndarray:
    """Free positions on a grid over the whole frame, nearest *anchor* first."""
    x0, y0, x1, y1 = frame
    step = max(1.0, h / 2)
    xs = np.arange(x0 + gap + w / 2, x1 - gap - w / 2 + 1e-9, step)
    ys = np.arange(y0 + gap + h / 2, y1 - gap - h / 2 + 1e-9, step)
    if not len(xs) or not len(ys):
        return np.empty((0, 2))
    grid = np.column_stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])
    free = grid[_clear(grid, w, h, placed, gap)]
    return free[np.argsort(((free - anchor) ** 2).sum(axis=1), kind="stable")]


def _first_acceptable(candidates, anchor, w, h, cover, risky, placed, leaders, lines: bool):
    """The first candidate that hides nothing and, if *lines*, crosses no leader
    line or label either way; None if there is none."""
    cover_w, cover_h = cover
    for start in range(0, len(candidates), _CHUNK):
        chunk = candidates[start:start + _CHUNK]
        keep = ~_hiding(chunk, cover_w, cover_h, risky) if risky else np.ones(len(chunk), dtype=bool)
        if lines and keep.any():
            keep[keep] = ~_crossing(anchor, chunk[keep], w, h, placed, leaders)
        if keep.any():
            return chunk[np.argmax(keep)]
    return None


def place_labels(anchors, sizes, frame, groups, *, hide: float = HIDE_FRACTION,
                 gap: float = 2.0) -> tuple[np.ndarray, int]:
    """Centres for label boxes that stay apart, inside the frame and off the groups.

    Everything is in pixels. ``anchors`` are where the labels belong (each group's
    median), ``sizes`` each box's width and height, ``frame`` the panel as
    ``(x0, y0, x1, y1)``, and ``groups`` the cells of each label's group.

    Labels are placed one at a time, largest group first. A position must lie
    inside the frame, *gap* from its edge, and keep *gap* from every label placed
    before it. It hides nothing if it covers at most *hide* of any group's cells
    and none of the other labels' anchors, counting anything up to *gap* outside
    the box as covered. A label over another's anchor would leave that one no
    leader line that misses it. Among the free positions, each label takes the
    first that exists of

    1. the nearest within two box lengths of its anchor that hides nothing, and
       needs no leader line across a label placed before it nor lies across one
       of their leader lines;
    2. the nearest such position anywhere in the frame;
    3. the nearest within reach that hides nothing, lines or not;
    4. the nearest at all.

    If no position is free, the label stays at its anchor, pulled inside the
    frame, and is counted as unplaced.

    Returns ``(centres, number unplaced)``.
    """
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
    sizes = np.asarray(sizes, dtype=float).reshape(-1, 2)
    centres = anchors.copy()
    table = _groups_at_risk(groups, hide)
    placed: list[tuple[float, float, float, float]] = []
    leaders: list[tuple[float, float, float, float]] = []
    unplaced = 0
    for i in sorted(range(len(anchors)), key=lambda i: (-len(groups[i]), i)):
        w, h = sizes[i]
        near = anchors[i] + _offsets(2 * (w + h), max(1.0, h / 4))
        near = near[_in_frame(near, w, h, frame, gap) & _clear(near, w, h, placed, gap)]
        # Cells count as covered up to *gap* outside the box, so none is left
        # sitting on its edge, where a pixel of rounding decides.
        cover_w, cover_h = w + 2 * gap, h + 2 * gap
        risky = [(cells, allowed) for cells, allowed, width, height in table
                 if width < cover_w and height < cover_h]
        if len(anchors) > 1:
            risky.append((np.delete(anchors, i, axis=0), 0))
        tests = (anchors[i], w, h, (cover_w, cover_h), risky, placed, leaders)

        centre = _first_acceptable(near, *tests, lines=True)
        if centre is None:
            anywhere = _free_grid(anchors[i], w, h, frame, placed, gap)
            centre = _first_acceptable(anywhere, *tests, lines=True)
            if centre is None:
                centre = _first_acceptable(near, *tests, lines=False)
            if centre is None and len(near):
                centre = near[0]
            if centre is None and len(anywhere):
                centre = anywhere[0]
        if centre is None:
            x0, y0, x1, y1 = frame
            centre = np.clip(anchors[i], [x0 + w / 2, y0 + h / 2], [x1 - w / 2, y1 - h / 2])
            unplaced += 1
        centres[i] = centre
        placed.append((centre[0], centre[1], w, h))
        end = _leader_end(anchors[i], centre, np.array([w, h]))
        if end is not None:
            leaders.append((anchors[i][0], anchors[i][1], end[0], end[1]))
    return centres, unplaced


def _box_size(text, renderer) -> tuple[float, float]:
    """A label's width and height in pixels, with the box drawn around it."""
    extent = text.get_window_extent(renderer)
    width, height = extent.width, extent.height
    patch = text.get_bbox_patch()
    if patch is not None:
        pad = getattr(patch.get_boxstyle(), "pad", 0.0) * renderer.points_to_pixels(text.get_fontsize())
        width, height = width + 2 * pad, height + 2 * pad
    return width, height


def _leader_end(anchor: np.ndarray, centre: np.ndarray, size: np.ndarray):
    """Where a line from *anchor* meets the label's box; None when none is needed.

    None when the anchor lies under the label, or closer to it than half the
    label's height: that close, the label still reads as its group's.
    """
    nearest = centre + np.clip(anchor - centre, -size / 2, size / 2)
    if np.hypot(*(anchor - nearest)) <= size[1] / 2:
        return None
    return nearest


class RepelledLabels(Artist):
    """Moves one panel's group labels apart each time the panel is drawn.

    The labels and their leader lines stay ordinary artists on the axes, found,
    styled and measured like any other; this artist only moves them. It draws
    nothing itself, so it takes no part in layout.
    """

    def __init__(self, texts, anchors, groups, leaders, gap_pt: float = GAP_PT):
        super().__init__()
        self._texts = list(texts)
        self._anchors = np.asarray(anchors, dtype=float).reshape(-1, 2)
        self._groups = [np.asarray(g, dtype=float).reshape(-1, 2) for g in groups]
        self._leaders = list(leaders)
        self._gap_pt = gap_pt
        self._warned = False
        self.set_in_layout(False)
        self.set_zorder(ZORDER)

    def draw(self, renderer):
        if not self.get_visible() or not self._texts:
            return
        ax = self.axes
        to_px = ax.transData
        from_px = to_px.inverted()
        anchors = to_px.transform(self._anchors)
        sizes = np.array([_box_size(text, renderer) for text in self._texts])
        centres, unplaced = place_labels(
            anchors, sizes, ax.bbox.extents, [to_px.transform(g) for g in self._groups],
            gap=renderer.points_to_pixels(self._gap_pt))

        # Artists only change when they must: every change marks the figure
        # stale, and an interactive backend answers that with another draw.
        for text, centre in zip(self._texts, centres):
            if np.abs(to_px.transform(text.get_position()) - centre).max() > 0.01:
                text.set_position(tuple(from_px.transform(centre)))
        for line, anchor, centre, size in zip(self._leaders, anchors, centres, sizes):
            end = _leader_end(anchor, centre, size)
            if end is None:
                if line.get_visible():
                    line.set_visible(False)
                continue
            ends = np.vstack([anchor, end])
            drawn = to_px.transform(np.asarray(line.get_xydata(), dtype=float).reshape(-1, 2))
            if drawn.shape != ends.shape or np.abs(drawn - ends).max() > 0.01:
                xy = from_px.transform(ends)
                line.set_data(xy[:, 0], xy[:, 1])
            if not line.get_visible():
                line.set_visible(True)

        if unplaced and not self._warned:
            self._warned = True
            warnings.warn(
                f"{unplaced} of {len(self._texts)} labels in this panel could not be "
                f"kept apart; give the figure more room or use a smaller label_size.",
                UserWarning, stacklevel=2)


def attach(ax, texts, anchors, groups, gap_pt: float = GAP_PT) -> RepelledLabels:
    """Hand *texts* to a :class:`RepelledLabels` on *ax*, with a leader line each.

    ``gap_pt`` is the space kept around each label. Where the groups are single
    points drawn as large markers, make it at least the marker's radius.
    """
    leaders = []
    for text in texts:
        # Placement keeps the labels inside the panel, so they can not widen the
        # layout. Measured before the first draw, at their anchors, they would.
        text.set_in_layout(False)
        line = Line2D([], [], color=text.get_color(), linewidth=0.6,
                      zorder=LEADER_ZORDER, visible=False)
        ax.add_line(line)
        leaders.append(line)
    artist = RepelledLabels(texts, anchors, groups, leaders, gap_pt)
    ax.add_artist(artist)
    return artist
