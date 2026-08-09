"""``diet_truecell`` — Seurat's ``DietSeurat``.

Strip an object down to the parts you still need: chosen assays, chosen layers
within them, a feature subset, and whichever reductions and graphs you name.
What it is for is making an object small enough to save, share or hold several
of at once — the counts and scale.data layers and the graphs are usually the
bulk of it, and all three are reproducible from what remains.
"""
from __future__ import annotations

import copy
import warnings
from typing import Optional, Union

from .assay import Assay
from .assay5 import StdAssay

__all__ = ["diet_truecell"]


def _slim_assay(assay, keep_layers: list[str]):
    """A copy of ``assay`` carrying only ``keep_layers``.

    Shallow throughout: the retained matrices are shared with the input rather
    than copied, since nothing here writes to them, and copying is the one thing
    a function whose whole purpose is to *free* memory must not do. The mutable
    bookkeeping is rebuilt so dropping a layer cannot reach back into the
    caller's object.
    """
    out = copy.copy(assay)
    out.layers = dict(assay.layers)
    out._layer_features = dict(assay._layer_features)
    out._layer_cells = dict(assay._layer_cells)
    out._split_stems = dict(assay._split_stems)
    out._cells = copy.copy(assay._cells)
    out._features = copy.copy(assay._features)
    out.misc = dict(assay.misc)
    for name in [n for n in assay.layers if n not in keep_layers]:
        out._drop_layer(name)
    # `default` is an *index* into the layer list, not a name, so dropping a
    # layer that sat before it silently re-points it at a different layer.
    # Resolve it to a name first and look that name up again.
    old_names = list(assay.layers)
    previous = old_names[min(assay.default, len(old_names) - 1)] if old_names else None
    kept = list(out.layers)
    out.default = kept.index(previous) if previous in kept else 0
    return out


def _resolve_layers(
    layers: Optional[Union[str, list[str], dict[str, Union[str, list[str]]]]],
    assays: list[str],
    objects: dict,
) -> dict[str, list[str]]:
    """Per-assay keep-lists, from R's ``.PropagateList`` + ``Layers(search=)``.

    A bare name or list applies to **every** assay; a dict names them
    individually. Names that an assay does not have are dropped rather than
    raising, so ``layers="scale.data"`` is usable across a mixed set of assays
    where only some carry one — R resolves the same way.
    """
    if layers is None:
        return {a: list(objects[a].layers) if isinstance(objects[a], StdAssay)
                else _v3_layers(objects[a]) for a in assays}
    if isinstance(layers, str):
        layers = [layers]
    if not isinstance(layers, dict):
        layers = {a: list(layers) for a in assays}

    out: dict[str, list[str]] = {}
    for assay in assays:
        want = layers.get(assay)
        if want is None:
            continue
        want = [want] if isinstance(want, str) else list(want)
        have = (list(objects[assay].layers) if isinstance(objects[assay], StdAssay)
                else _v3_layers(objects[assay]))
        kept = [n for n in have if n in want]
        if kept:
            out[assay] = kept
    return out


def _v3_layers(assay: Assay) -> list[str]:
    """The layer names a v3 ``Assay`` exposes, in Seurat's order."""
    return ["counts", "data", "scale.data"]


def diet_truecell(
    seurat,
    layers: Optional[Union[str, list[str], dict[str, Union[str, list[str]]]]] = None,
    features: Optional[list[str]] = None,
    assays: Optional[Union[str, list[str]]] = None,
    dimreducs: Optional[Union[str, list[str]]] = None,
    graphs: Optional[Union[str, list[str]]] = None,
    misc: bool = True,
):
    """Slim an object down to the pieces you name.

    Mirrors R's ``DietSeurat(object)``. Returns a **new** object; the one passed
    in is left alone, and the layers that survive are shared rather than copied,
    so this frees memory rather than doubling it.

    Parameters
    ----------
    layers    : layers to keep. A name or list applies to every assay; a dict
                (``{"RNA": "counts"}``) names them per assay. ``None`` keeps all.
    features  : keep only these features. An assay left with none of them is
                dropped, with a warning.
    assays    : assays to keep. ``None`` keeps all.
    dimreducs : reductions to keep. **``None`` keeps none** — see below.
    graphs    : graphs to keep. **``None`` keeps none** — see below.
    misc      : ``False`` empties the object-level ``misc``.

    Returns
    -------
    A new ``Truecell``.

    Warning
    -------
    **``diet_truecell(obj)`` with no arguments deletes every reduction and every
    graph.** That is R's behaviour, not a translation slip — ``dimreducs`` and
    ``graphs`` are keep-lists, and an unset keep-list keeps nothing. Verified
    against Seurat 5.5.1: a pbmc3k object with a `pca` and two graphs comes back
    with zero of each, and all three layers untouched. Name what you want kept::

        slim = diet_truecell(obj, layers="counts", dimreducs="pca")

    Notes
    -----
    - **Neighbor objects are left alone**, as in R: ``DietSeurat`` filters only
      ``DimReduc`` and ``Graph``.
    - A v3 ``Assay`` cannot give up both ``counts`` and ``data`` — Seurat raises
      rather than leave the assay with no expression matrix, and so does this.
    - Cells are re-intersected against the surviving assays at the end, which
      only bites when ``features`` empties an assay and removes it.
    """
    available = list(seurat.assays)
    if assays is None:
        keep_assays = list(available)
    else:
        want = [assays] if isinstance(assays, str) else list(assays)
        keep_assays = [a for a in available if a in want]
    if not keep_assays:
        raise ValueError("No assays provided were found in the Truecell object")
    if seurat.active_assay not in keep_assays:
        raise ValueError(
            f"The default assay ({seurat.active_assay!r}) is slated to be "
            f"removed, please change the default assay"
        )

    keep_layers = _resolve_layers(layers, keep_assays, seurat.assays)
    if not keep_layers:
        raise ValueError("None of the requested layers found")

    new_assays = {}
    for name in keep_assays:
        assay = seurat.assays[name]
        wanted = keep_layers.get(name)
        if not wanted:
            continue
        if isinstance(assay, Assay):
            # v3 keeps its matrices in fixed slots, so "removing" a layer means
            # emptying it; losing both counts and data would leave nothing to
            # express the assay with, which R refuses rather than allows.
            if "counts" not in wanted and "data" not in wanted:
                raise ValueError("Cannot remove both 'counts' and 'data' from v3 Assays")
            slim = copy.copy(assay)
            slim.misc = dict(assay.misc)
            for slot, attr in (("counts", "counts"), ("data", "data"),
                               ("scale.data", "scale_data")):
                if slot not in wanted:
                    setattr(slim, attr, _empty_like(getattr(assay, attr)))
        else:
            slim = _slim_assay(assay, wanted)

        if features is not None:
            present = [f for f in features if f in set(slim.features())]
            if not present:
                warnings.warn(
                    f"No features found in assay {name!r}, removing...",
                    stacklevel=2,
                )
                continue
            slim = slim.subset(features=present)
        new_assays[name] = slim

    if not new_assays:
        raise ValueError("No assays survived; nothing left to diet")
    if seurat.active_assay not in new_assays:
        raise ValueError(
            f"The default assay ({seurat.active_assay!r}) was removed by the "
            f"`features` filter, please change the default assay"
        )

    def _keep(requested, have):
        if requested is None:
            return {}
        want = [requested] if isinstance(requested, str) else list(requested)
        return {k: v for k, v in have.items() if k in want}

    out = seurat.__class__(
        assays=new_assays,
        meta_data=seurat.meta_data.copy(),
        active_assay=seurat.active_assay,
        active_ident=seurat._active_ident,
        graphs=_keep(graphs, seurat.graphs),
        neighbors=dict(seurat.neighbors),
        reductions=_keep(dimreducs, seurat.reductions),
        images=dict(seurat.images),
        project_name=seurat.project_name,
        misc=dict(seurat.misc) if misc else {},
        version=seurat.version,
        commands=list(seurat.commands),
        tools=dict(seurat.tools),
    )

    # R re-intersects the object's cells against what the assays still hold.
    # Only `features` emptying an assay can move this, but when it does the
    # metadata and idents have to follow the assays rather than lead them.
    surviving = set()
    for assay in new_assays.values():
        surviving.update(assay.cells())
    kept_cells = [c for c in out.cell_names() if c in surviving]
    if len(kept_cells) < len(out.cell_names()):
        out = out.subset(cells=kept_cells)
    return out


def _empty_like(mat):
    """An empty stand-in with the same type, for a v3 slot being cleared."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(mat):
        return sp.csc_matrix((0, mat.shape[1]), dtype=mat.dtype)
    return np.zeros((0, mat.shape[1]), dtype=np.asarray(mat).dtype)
