"""``truecell.generics`` declares only what something implements.

Thirty-nine of its seventy-two generics had only the ``NotImplementedError``
fallback registered, so each raised for every type of object, and
``skills/truecell/reference/api-map.md`` listed many of them as the way to do
things (``set_ident``, ``set_variable_features``, ``hvf_info``, …). Nothing walked
the module, and the api-map's list is prose, so no test read it either.

A generic stays where the object has a method, a property or a log behind it, and
delegates to that, which is the "same code path" ``docs/api/generics.md`` promises.
The ones with nothing behind them, or with a different meaning from R's, are gone.
"""
import importlib
import pathlib
import pkgutil
import re

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import truecell
from truecell import generics as g

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _import(name):
    try:
        return importlib.import_module(name)
    except ImportError:                                # an extra that is not installed
        return None


def _generics():
    for module in pkgutil.walk_packages(truecell.__path__, "truecell."):
        _import(module.name)                           # so that every @register has run
    return {n: f for n, f in vars(g).items() if not n.startswith("_") and hasattr(f, "registry")}


# ---------------------------------------------------------------------------
# What is declared is implemented, and what is documented is declared
# ---------------------------------------------------------------------------

def test_every_generic_has_an_implementation():
    stubs = sorted(name for name, generic in _generics().items()
                   if all(getattr(impl, "_is_stub", False) for impl in generic.registry.values()))
    assert not stubs, f"declared, but implemented for no type: {stubs}"


def test_the_fallback_is_what_marks_a_stub():
    """The check above reads `_is_stub`, so it must be on the fallback and only there."""
    fallback = g.cells.registry[object]
    assert fallback._is_stub is True
    assert not any(getattr(impl, "_is_stub", False)
                   for impl in g.cells.registry.values() if impl is not fallback)


def _backticked(text):
    return set(re.findall(r"`([a-z][a-z_0-9]*)`", text))


def test_the_api_maps_generics_are_all_declared():
    """The list is prose, so `test_api_map_signatures` cannot read it."""
    text = (ROOT / "skills/truecell/reference/api-map.md").read_text()
    section = text.split("## Generics", 1)[1].split("\n## ", 1)[0]
    listed = _backticked(section)
    assert len(listed) > 40                            # the section was found, not an empty one
    assert not listed - set(_generics()), "listed as a generic, but not in truecell.generics"
    assert not set(_generics()) - listed, "a generic the api-map does not list"


def test_the_generics_the_skill_names_are_declared():
    text = (ROOT / "skills/truecell/SKILL.md").read_text()
    paragraph = text.split("**6. The generics are not top-level.**", 1)[1].split("```", 1)[0]
    named = _backticked(paragraph) - {"truecell"}
    assert len(named) > 10
    assert not named - set(_generics())


@pytest.mark.parametrize("name", [
    "as_segmentation", "as_seurat", "assay_class", "check_matrix", "create_assay_object",
    "create_centroids", "create_fov", "create_segmentation", "create_truecell_object",
    "default_dim_reduc", "default_fov", "hvf_info", "keys", "list_to_s4", "match_cells",
    "s4_to_list",
])
def test_a_generic_with_nothing_behind_it_is_gone(name):
    """Not the constructors either: `truecell.create_fov(coords=...)` takes its input by
    keyword, which a dispatcher on the first positional argument cannot."""
    assert not hasattr(g, name)


# ---------------------------------------------------------------------------
# Each delegate is the object's own method, property or log
# ---------------------------------------------------------------------------

@pytest.fixture
def obj():
    """The standard workflow on 60 genes and 30 cells, with an ADT assay beside RNA."""
    rng = np.random.default_rng(0)
    o = truecell.create_truecell_object(
        counts=sp.csc_matrix(rng.poisson(2.0, size=(60, 30)).astype(float)),
        feature_names=[f"g{i}" for i in range(60)],
        cell_names=[f"c{j}" for j in range(30)],
    )
    truecell.normalize_data(o)
    truecell.find_variable_features(o, nfeatures=20)
    truecell.scale_data(o)
    truecell.run_pca(o, n_pcs=5)
    o.assays["ADT"] = truecell.create_assay5_object(
        counts=sp.csc_matrix(rng.poisson(5.0, size=(4, 30)).astype(float)),
        feature_names=list("abcd"), cell_names=o.cell_names(), key="adt_")
    return o


def test_assay_names_and_version(obj):
    assert g.assay_names(obj) == obj.assay_names() == ["RNA", "ADT"]
    assert g.version(obj) == obj.version


def test_set_default_assay_goes_through_the_property_and_its_check(obj):
    g.set_default_assay(obj, "ADT")
    assert obj.default_assay == "ADT"
    with pytest.raises(KeyError, match="not found"):
        g.set_default_assay(obj, "nope")
    reduction = obj.reductions["pca"]                  # reductions carry their own
    g.set_default_assay(reduction, "ADT")
    assert g.default_assay(reduction) == "ADT"


def test_set_ident_and_reorder_ident(obj):
    g.set_ident(obj, ["c0", "c1"], "special")
    assert list(obj.idents[:2]) == ["special", "special"]

    obj.idents = pd.Categorical(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    obj.meta_data["score"] = -np.arange(30, dtype=float)       # C has the lowest mean, then B
    g.reorder_ident(obj, "score")
    assert list(obj.idents.categories) == ["C", "B", "A"]


def test_tool_and_set_tool(obj):
    g.set_tool(obj, "k", {"a": 1})
    assert g.tool(obj, "k") == {"a": 1} == obj.tool("k") == obj.tools["k"]


def test_misc_and_set_misc_on_every_container(obj):
    g.set_misc(obj, 3, "n")                            # Misc(x, slot) <- value
    assert g.misc(obj, "n") == 3 == obj.misc["n"]
    g.set_misc(obj, {"z": 1})                          # Misc(x) <- value replaces the list
    assert g.misc(obj) is obj.misc and obj.misc == {"z": 1}
    for part in (obj.get_assay(), obj.reductions["pca"]):
        g.set_misc(part, "v", "slot")
        assert g.misc(part, "slot") == "v" == part.misc["slot"]


def test_set_variable_features_for_an_object_and_for_its_assay(obj):
    assay = obj.get_assay()
    g.set_variable_features(obj, ["g3", "g1", "g2"])
    assert g.variable_features(obj) == assay.variable_features == ["g3", "g1", "g2"]
    g.set_variable_features(assay, ["g5"])
    assert assay.variable_features == ["g5"]
    g.set_variable_features(obj, ["a", "b"], assay="ADT")
    assert obj.get_assay("ADT").variable_features == ["a", "b"]
    assert assay.variable_features == ["g5"]           # the other assay is left alone


def test_set_key_and_set_default_layer(obj):
    assay, reduction = obj.get_assay(), obj.reductions["pca"]
    g.set_key(assay, "newkey_")
    g.set_key(reduction, "PCX_")
    assert g.key(assay) == assay.key == "newkey_" and reduction.key == "PCX_"
    g.set_default_layer(assay, "data")
    assert assay.default_layer == "data"
    with pytest.raises(KeyError, match="not found"):
        g.set_default_layer(assay, "no_such_layer")


def test_layer_setters_and_calc_n(obj):
    assay = obj.get_assay()
    doubled = assay.layer_data("counts") * 2
    g.set_layer_data(assay, "doubled", doubled)
    g.set_assay_data(assay, "also_doubled", doubled)   # SetAssayData on a v5 assay is LayerData<-
    for name in ("doubled", "also_doubled"):
        assert (assay.layer_data(name) != doubled).nnz == 0
    pd.testing.assert_frame_equal(g.calc_n(assay), assay.calc_n())
    assert g.cast_assay(assay).layers.keys() == assay.cast_assay().layers.keys()


def test_set_assay_data_on_a_v3_assay():
    rng = np.random.default_rng(1)
    o = truecell.create_truecell_object(
        counts=sp.csc_matrix(rng.poisson(2.0, size=(10, 6)).astype(float)),
        feature_names=[f"g{i}" for i in range(10)], cell_names=list("uvwxyz"), use_v5=False)
    assay = o.get_assay()
    new = sp.csc_matrix(np.ones((10, 6)))
    g.set_assay_data(assay, "data", new)
    assert (assay.get_assay_data("data") != new).nnz == 0
    pd.testing.assert_frame_equal(g.calc_n(assay), assay.calc_n())


def test_set_loadings(obj):
    reduction = obj.reductions["pca"]
    doubled = g.loadings(reduction) * 2
    g.set_loadings(reduction, doubled)
    np.testing.assert_array_equal(g.loadings(reduction), doubled)


# ---- the command log ----

def test_command_lists_what_has_run_and_reads_one_back(obj):
    assert {"NormalizeData.RNA", "FindVariableFeatures.RNA", "ScaleData.RNA"} <= set(g.command(obj))
    entry = g.command(obj, "FindVariableFeatures.RNA")
    assert entry is obj.commands[[c.key for c in obj.commands].index("FindVariableFeatures.RNA")]
    assert g.command(obj, "FindVariableFeatures.RNA", "nfeatures") == 20


def test_command_errors_as_r_does(obj):
    with pytest.raises(KeyError, match="has not been run or is not a valid command"):
        g.command(obj, "NoSuchCommand")
    # `value` reads the parameters only: R checks `names(params)`, not the slots
    with pytest.raises(KeyError, match="is not a valid parameter for FindVariableFeatures"):
        g.command(obj, "FindVariableFeatures.RNA", "name")


def test_a_command_run_twice_keeps_one_entry_holding_the_latest(obj):
    """`obj@commands` is a named list in R, so a second `FindVariableFeatures` replaces
    the first in its place; `obj.commands` is a list that grows."""
    truecell.find_variable_features(obj, nfeatures=10)
    keys = [c.key for c in obj.commands]
    assert keys.count("FindVariableFeatures.RNA") == 2
    names = g.command(obj)
    assert names.count("FindVariableFeatures.RNA") == 1
    assert names.index("FindVariableFeatures.RNA") == keys.index("FindVariableFeatures.RNA")
    assert g.command(obj, "FindVariableFeatures.RNA", "nfeatures") == 10


# ---- spatial ----

@pytest.fixture
def fov():
    rng = np.random.default_rng(7)
    cells = [f"c{i}" for i in range(20)]
    coords = pd.DataFrame({"x": rng.uniform(0, 100, 20), "y": rng.uniform(0, 100, 20), "cell": cells})
    molecules = pd.DataFrame({
        "x": rng.uniform(0, 100, 60), "y": rng.uniform(0, 100, 60),
        "gene": [f"m{i % 3}" for i in range(60)], "cell": [cells[i % 20] for i in range(60)]})
    f = truecell.create_fov(coords, type_="centroids")
    f.molecules = {"rna": truecell.create_molecules(molecules)}
    return f


def test_fov_delegates(fov):
    assert g.default_boundary(fov) == fov.default_boundary() is not None
    cropped = g.crop(fov, (0, 50), (0, 50))
    assert cropped.cells() == fov.crop((0, 50), (0, 50)).cells()
    assert 0 < len(cropped.cells()) < len(fov.cells())          # it cropped something
    assert g.overlay(fov, cropped) == fov.overlay(cropped)
    assert g.get_molecules(fov, "rna") is fov.get_molecules("rna")
    assert g.get_molecules(fov) == fov.get_molecules()


def test_a_visium_image_is_an_fov_so_it_dispatches_the_same():
    from truecell.spatial.fov import FOV
    from truecell.spatial.visium import VisiumV2
    for generic in (g.crop, g.overlay, g.default_boundary, g.get_molecules):
        assert generic.dispatch(VisiumV2) is generic.dispatch(FOV)
        assert not getattr(generic.dispatch(VisiumV2), "_is_stub", False)


def test_as_centroids_of_a_segmentation():
    rows = [{"x": float(x), "y": float(y), "cell": c}
            for c in ("a", "b") for x, y in ((0, 0), (4, 0), (4, 4), (0, 4))]
    segmentation = truecell.create_segmentation(pd.DataFrame(rows))
    pd.testing.assert_frame_equal(g.as_centroids(segmentation).get_tissue_coordinates(),
                                  segmentation.as_centroids().get_tissue_coordinates())
