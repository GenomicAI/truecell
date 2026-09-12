"""Every default a ported function shares with Seurat must match Seurat's, or say why not.

`tools/compare_defaults.py` does the comparison. This runs it against
`tests/data/seurat_formals.json`, which `python tools/compare_defaults.py
--refresh` dumps from a live Seurat, so nothing here needs R.

`find_markers` shipped Seurat 4's thresholds for months after Seurat 5 changed
them, and it was the second such miss to be found by accident. A default reads
correctly in a diff and nobody re-derives it; this is the check that does.
"""
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "compare_defaults", ROOT / "tools" / "compare_defaults.py")
cd = importlib.util.module_from_spec(_spec)
# Registered before it runs: `@dataclass` resolves the module's annotations
# through sys.modules, and a module loaded from a path is not in it otherwise.
sys.modules[_spec.name] = cd
_spec.loader.exec_module(cd)


@pytest.fixture(scope="module")
def reference():
    return cd.load_reference()


@pytest.fixture(scope="module")
def comparison(reference):
    return cd.compare(reference)


def test_the_reference_is_the_seurat_the_port_targets(reference):
    assert (reference["seurat"], reference["seurat_object"]) == ("5.5.1", "5.4.0")


def test_every_mapped_function_exists_on_both_sides(reference):
    assert cd.missing_functions(reference) == []


def test_no_default_differs_from_seurats_without_a_reason(comparison):
    mismatches, _ = comparison
    unexplained = cd.unexplained(mismatches)
    assert not unexplained, (
        "defaults that differ from Seurat's with no reason written down — fix the "
        "default, or add an entry to tools/compare_defaults.py saying why:\n"
        + "\n".join(f"  {m}" for m in unexplained))


def test_no_listed_reason_outlives_the_difference_it_explains(comparison):
    mismatches, _ = comparison
    stale = cd.stale(mismatches)
    assert not stale, f"entries that no longer describe a live difference, delete them: {stale}"


def test_every_reason_says_something():
    for key, why in {**cd.EQUIVALENT, **cd.KNOWN_DIVERGENCES}.items():
        assert len(why) > 30, f"{key}: a reason has to be a reason"


def test_the_comparison_is_not_vacuous(comparison):
    """A parser that evaluated nothing would report no mismatch and pass the above."""
    mismatches, stats = comparison
    assert stats["compared"] >= 300
    # A difference that stays on purpose, so this does not go stale as fixes land.
    assert ("truecell.dim_plot", "pt_size") in {m.key for m in mismatches}


@pytest.mark.parametrize("name", ["find_markers", "find_all_markers", "find_conserved_markers"])
def test_marker_thresholds_are_seurat_5s(name):
    """The miss that prompted the guard, pinned by name too. R: `formals(FindMarkers.default)`
    and `formals(FindAllMarkers)` give logfc.threshold = 0.1 and min.pct = 0.01."""
    import truecell

    params = inspect.signature(getattr(truecell, name)).parameters
    assert params["logfc_threshold"].default == 0.1
    assert params["min_pct"].default == 0.01


def test_a_deprecated_alias_does_not_stand_in_for_the_formal_it_renames(monkeypatch):
    """FindSpatiallyVariableFeatures.Seurat declares `layer = "scale.data"` and the
    deprecated `slot = NULL`. Both map to truecell's `layer`; the real formal decides."""
    name = "truecell.find_spatially_variable_features"
    reference = cd.load_reference()
    spec = cd.FUNCTIONS[name][0]
    assert {"layer", "slot"} <= set(reference["functions"][spec])
    monkeypatch.setattr(cd, "FUNCTIONS", {name: cd.FUNCTIONS[name]})
    mismatches, stats = cd.compare(reference)
    assert stats["compared"] > 0
    assert (name, "layer") not in {m.key for m in mismatches}
