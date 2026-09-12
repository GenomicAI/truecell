"""select_integration_features against Seurat 5.5.1's SelectIntegrationFeatures.

`tests/data/r_select_integration_features.json` is written by
`tests/data/make_select_integration_features_reference.R`: three hand-built
variable-feature lists, run at nfeatures values that put the cut inside ties on
how many objects call a gene variable and on its median rank.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from truecell import create_truecell_object, select_integration_features

REF = json.loads(
    (Path(__file__).parent / "data" / "r_select_integration_features.json").read_text())


def _object(features, variable):
    obj = create_truecell_object(counts=sp.csc_matrix(np.ones((len(features), 3))),
                                 feature_names=list(features), cell_names=["a", "b", "c"])
    obj.get_assay().variable_features = list(variable)
    return obj


def _objects():
    return [_object(f, v) for f, v in zip(REF["features"], REF["variable_features"])]


@pytest.mark.parametrize("case", REF["cases"], ids=lambda c: f"nfeatures{c['nfeatures']}")
def test_matches_seurat(case):
    assert select_integration_features(_objects(), nfeatures=case["nfeatures"]) == case["selected"]


def test_the_reference_reaches_every_tie_it_is_there_for():
    """The cut has to land inside a tie on count and inside one on median rank."""
    by_n = {c["nfeatures"]: c["selected"] for c in REF["cases"]}
    assert by_n[7][-1] == "F11" and by_n[8][-1] == "F13"   # tied count and rank: name decides
    assert "F29" not in by_n[20]                          # absent from the first object


def test_a_list_every_object_shares_comes_back_in_its_own_order():
    same = REF["identical_lists"]
    genes = [f"F{i:02d}" for i in range(1, 31)]
    objs = [_object(genes, same["variable_features"]) for _ in range(2)]
    assert select_integration_features(objs) == same["selected"]


def test_a_full_tie_breaks_by_name_in_code_point_order():
    """R breaks it in table()'s collation order; Rscript runs LC_COLLATE=C, code-point order."""
    mixed = REF["mixed_case"]
    assert REF["collate"] == "C"
    objs = [_object(mixed["features"], v) for v in mixed["variable_features"]]
    assert select_integration_features(objs, nfeatures=mixed["nfeatures"]) == mixed["selected"]


def test_find_integration_anchors_chooses_2000_features_by_default(monkeypatch):
    """FindIntegrationAnchors(anchor.features = 2000) runs SelectIntegrationFeatures."""
    from truecell import anchors

    seen = {}

    def stop(objects, nfeatures=2000, **kwargs):
        seen["nfeatures"] = nfeatures
        raise RuntimeError("selected")

    monkeypatch.setattr(anchors, "select_integration_features", stop)
    with pytest.raises(RuntimeError, match="selected"):
        anchors.find_integration_anchors(_objects())
    assert seen["nfeatures"] == 2000


def test_an_object_without_variable_features_is_not_changed_by_selection():
    """Seurat computes them on its own copy of the object; so does this."""
    rng = np.random.default_rng(0)
    genes = [f"F{i:02d}" for i in range(1, 31)]
    bare = create_truecell_object(counts=sp.csc_matrix(rng.poisson(3.0, size=(30, 40)).astype(float)),
                                  feature_names=genes, cell_names=[f"c{i}" for i in range(40)])
    from truecell.preprocessing import normalize_data

    normalize_data(bare)
    assert bare.get_assay().variable_features == []
    selected = select_integration_features([bare, _objects()[1]], nfeatures=10)
    assert len(selected) == 10
    assert bare.get_assay().variable_features == []
