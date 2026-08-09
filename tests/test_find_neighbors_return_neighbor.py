"""`find_neighbors(return_neighbor=True)` — Seurat's `return.neighbor`.

`find_neighbors` used to take an `nn_name` argument that was never read, and
nothing in truecell ever populated `seurat.neighbors`. Seurat has no `nn.name`
at all; what it has is `return.neighbor`, which stores the raw KNN — indices and
distances — as a `Neighbor` instead of building graphs. The distances were
already being computed and thrown away.

The reference in ``data/find_neighbors_r_reference.json`` is a live Seurat 5.5.1
run on pbmc3k with ``nn.method = "rann"`` (exact, matching sklearn; the default
``annoy`` is approximate and would not be a fair reference). It carries R's own
PCA embedding so the comparison isolates the neighbour search from truecell's
PCA. On the full 2,700 cells every one of the 54,000 indices matched and the
distances agreed to 1.1e-13.

The fixture keeps the union of 25 seed cells' neighbourhoods rather than a
contiguous block: dropping cells that were *farther* than a seed's 20 nearest
cannot change which 20 are nearest, so the seeds' answers are still R's.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import truecell as tc
from truecell import DimReduc, Neighbor, create_truecell_object

REFERENCE = Path(__file__).parent / "data" / "find_neighbors_r_reference.json"
K = 20


@pytest.fixture
def r_object():
    """An object carrying R's pbmc3k PCA embedding, and R's expected answers."""
    ref = json.loads(REFERENCE.read_text())
    cells = ref["cells"]
    emb = np.array(ref["pca"], dtype=float)
    obj = create_truecell_object(
        sp.csc_matrix((5, len(cells))), assay="RNA",
        min_cells=0, min_features=0,
        feature_names=[f"g{i}" for i in range(5)], cell_names=cells)
    obj.reductions["pca"] = DimReduc(
        cell_embeddings=emb, cell_names=cells, key="PC_", assay_used="RNA")
    return obj, ref


# ---------------------------------------------------------------------------
# Parity with Seurat
# ---------------------------------------------------------------------------

def test_indices_and_distances_match_seurat(r_object):
    """R's `Indices()`/`Distances()`, cell for cell.

    Indices are compared exactly — a nearest-neighbour list is a set of integers
    and there is no tolerance to spend on it. Distances get a float tolerance
    because both sides take a square root, nothing more.
    """
    obj, ref = r_object
    tc.find_neighbors(obj, dims=range(10), k_param=K, return_neighbor=True)
    nn = obj.neighbors["RNA.nn"]

    rows = ref["check_rows"]
    got_i = nn.indices()[rows]
    got_d = nn.distances()[rows]
    want_i = np.array(ref["expected_idx"], dtype=int)
    want_d = np.array(ref["expected_dist"], dtype=float)

    assert got_i.shape == want_i.shape == (len(rows), K)
    assert np.array_equal(got_i, want_i), (
        f"{int((got_i != want_i).sum())} of {got_i.size} neighbour indices "
        f"differ from Seurat"
    )
    assert np.abs(got_d - want_d).max() < 1e-10


def test_indices_are_zero_based_where_r_is_one_based(r_object):
    """A convention difference, pinned so it cannot drift into an off-by-one.

    R's `Indices()` are 1-based; truecell's are 0-based, like every other index
    it exposes. The self-neighbour is what makes this checkable without the
    reference: cell i's nearest neighbour is itself, so column 0 is `arange(n)`
    here and would be `arange(n) + 1` in R.
    """
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10), k_param=K, return_neighbor=True)
    idx = obj.neighbors["RNA.nn"].indices()
    assert np.array_equal(idx[:, 0], np.arange(idx.shape[0]))
    assert idx.min() == 0


def test_self_is_first_at_distance_zero(r_object):
    """Seurat's `k.param` counts the cell itself, so k=20 means 19 others."""
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10), k_param=K, return_neighbor=True)
    nn = obj.neighbors["RNA.nn"]
    assert nn.dim() == (len(obj.cell_names()), K)
    assert np.allclose(nn.distances()[:, 0], 0.0)
    # Distances are non-decreasing across the row.
    d = nn.distances()
    assert (np.diff(d, axis=1) >= -1e-12).all()


# ---------------------------------------------------------------------------
# What gets stored, and what does not
# ---------------------------------------------------------------------------

def test_return_neighbor_stores_a_neighbor_and_no_graphs(r_object):
    """Seurat's two modes are exclusive: a Neighbor *instead of* graphs."""
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10), return_neighbor=True)
    assert list(obj.neighbors) == ["RNA.nn"]
    assert isinstance(obj.neighbors["RNA.nn"], Neighbor)
    assert obj.graphs == {}


def test_default_still_stores_the_two_graphs_and_no_neighbor(r_object):
    """The existing behaviour must not move; this is the path everything uses."""
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10))
    assert sorted(obj.graphs) == ["RNA_nn", "RNA_snn"]
    assert obj.neighbors == {}


def test_the_neighbor_name_uses_a_dot_not_an_underscore(r_object):
    """`RNA.nn` against `RNA_nn` / `RNA_snn` — Seurat's naming, verified.

    The separator is the only thing distinguishing a stored Neighbor from the
    KNN graph in a printout, so getting it wrong is quiet and confusing.
    """
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10), return_neighbor=True)
    assert "RNA.nn" in obj.neighbors
    assert "RNA_nn" not in obj.neighbors


def test_graph_name_names_the_neighbor_outright(r_object):
    """With `return.neighbor`, R's `graph.name` is the name, not a prefix."""
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10), return_neighbor=True,
                      graph_name="custom.nn")
    assert list(obj.neighbors) == ["custom.nn"]


# ---------------------------------------------------------------------------
# compute_snn
# ---------------------------------------------------------------------------

def test_compute_snn_false_skips_the_snn_graph(r_object):
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10), compute_snn=False)
    assert list(obj.graphs) == ["RNA_nn"]


def test_asking_for_both_warns_and_computes_no_snn(r_object):
    """R warns rather than refusing: 'The SNN graph is not computed if
    return.neighbor is TRUE.'"""
    obj, _ = r_object
    with pytest.warns(UserWarning, match="SNN graph is not computed"):
        tc.find_neighbors(obj, dims=range(10), return_neighbor=True,
                          compute_snn=True)
    assert obj.graphs == {}
    assert list(obj.neighbors) == ["RNA.nn"]


def test_the_command_log_records_the_mode(r_object):
    """`obj.commands` is what tells you which mode produced the object."""
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10), return_neighbor=True)
    last = obj.commands[-1]
    assert last.params["return_neighbor"] is True
    assert last.params["compute_snn"] is False


def test_neighbor_survives_a_rename(r_object):
    """`Neighbor.rename_cells` exists; nothing produced one to exercise it."""
    obj, _ = r_object
    tc.find_neighbors(obj, dims=range(10), return_neighbor=True)
    nn = obj.neighbors["RNA.nn"]
    renamed = nn.rename_cells([f"x_{c}" for c in nn.cells()])
    assert renamed.cells()[0].startswith("x_")
    assert np.array_equal(renamed.indices(), nn.indices())
