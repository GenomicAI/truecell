"""Seurat's modularity optimiser, ported: partitions against Seurat 5.5.1, label for label.

``find_clusters`` with algorithm 1, 2 or 3 runs ``truecell/_modularity.py``, a translation
of the C++ behind Seurat's ``RunModularityClusteringCpp``. Every expected partition here
was written from a live Seurat by ``tests/data/make_modularity_reference.R``. Each test
asserts exact equality, because the port claims the same partition, not a similar one:
the same ``java.util.Random`` stream, the same order of visiting nodes and the same
restarts.

Seurat's own partition depends on how its C++ was compiled. An arm64 build fuses the
multiply-add in the rule that moves a node, which flips a move whose gain is within one
rounding of zero; an x86_64 build does not. The reference holds only graphs on which the
two builds agree, and truecell follows the x86_64 arithmetic
(``test_a_move_is_decided_in_plain_ieee_arithmetic``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("numba")

from truecell import _modularity as modularity
from truecell import clustering, create_truecell_object, find_clusters
from truecell.clustering import _group_singletons, _r_sample_index
from truecell.graph import Graph

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "r_modularity_reference.json").read_text())


def _snn():
    """The FindNeighbors graph; its weights are inter / (2k - inter), stored as counts."""
    g = REFERENCE["snn"]
    inter = np.asarray(g["inter"], dtype=float)
    return sp.csc_matrix((inter / (2 * g["k"] - inter), g["i"], g["p"]), shape=(g["n"], g["n"]))


def _asym():
    g = REFERENCE["asym"]
    return sp.csc_matrix((np.asarray(g["x"], dtype=float), g["i"], g["p"]),
                         shape=(g["n"], g["n"]))


def _circulant(n):
    """Each node linked to the next four, with weights R computes the same way."""
    i = np.repeat(np.arange(n), 4)
    d = np.tile(np.arange(1, 5), n)
    j = (i + d) % n
    w = np.sqrt(1 + (7 * i + d) % 97)
    return sp.csc_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                         shape=(n, n))


GRAPHS = {"snn": _snn, "asym": _asym}
RUNS = [(graph, run) for graph in GRAPHS for run in REFERENCE[graph]["runs"]]


def _run_id(graph, run):
    return (f"{graph}-alg{run['algorithm']}-mf{run['modularity_fxn']}-seed{run['random_seed']}"
            f"-{run['n_start']}x{run['n_iter']}")


@pytest.fixture(scope="module")
def networks():
    return {(graph, mf): modularity.build_network(make(), mf)
            for graph, make in GRAPHS.items() for mf in (1, 2)}


# ---------------------------------------------------------------------------
# The optimiser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("graph,run", RUNS, ids=[_run_id(g, r) for g, r in RUNS])
def test_the_partition_is_seurats(networks, graph, run):
    """Algorithms 1-3, both modularity functions, two seeds, one start and ten.

    ``asym`` is not symmetric, so it also pins which triangle Seurat reads.
    """
    labels = modularity.run_modularity_clustering(
        networks[graph, run["modularity_fxn"]], run["resolution"], run["algorithm"],
        run["n_start"], run["n_iter"], run["random_seed"], run["modularity_fxn"])
    np.testing.assert_array_equal(labels, run["labels"])


@pytest.mark.parametrize("run", REFERENCE["rejection"]["runs"],
                         ids=lambda run: f"seed{run['random_seed']}")
def test_a_seed_whose_permutation_redraws_is_seurats(run):
    """``nextInt`` redraws a value that would bias it, and Seurat's build keeps that step.

    On 3,000 nodes, seed 38 first redraws at draw 820 of the first node permutation, and
    dropping the redraw changes the partition. Seed 0 does not redraw there.
    """
    rejection = REFERENCE["rejection"]
    network = modularity.build_network(_circulant(rejection["n"]), 1)
    labels = modularity.run_modularity_clustering(network, rejection["resolution"], 1, 1, 1,
                                                  run["random_seed"], 1)
    np.testing.assert_array_equal(labels, run["labels"])


def test_java_random_is_java_util_random():
    """Draws pinned to Seurat's own ``JavaRandom``, compiled from its C++ source."""
    np.testing.assert_array_equal(
        modularity.next_ints(0, 100000, 12),
        [41360, 5948, 48029, 16447, 43515, 81053, 54491, 69761, 18719, 32854, 81077, 32677])
    # Seurat takes the seed as a C int, so -1 is sign-extended before it is scrambled.
    np.testing.assert_array_equal(modularity.next_ints(-1, 1000, 8),
                                  [913, 225, 579, 439, 604, 438, 477, 478])
    # Draw 1,639 (from 0) is the first that nextInt(100000) redraws from seed 0; without
    # the redraw it and every later draw would shift by one.
    np.testing.assert_array_equal(modularity.next_ints(0, 100000, 1642)[1638:],
                                  [84456, 30233, 19992, 57005])
    # A power of two takes the top bits instead of a remainder.
    np.testing.assert_array_equal(modularity.next_ints(0, 16, 8), [11, 13, 3, 9, 10, 4, 8, 1])
    np.testing.assert_array_equal(modularity.next_ints(42, 1048576, 6),
                                  [762905, 57320, 716411, 50268, 323715, 987835])
    np.testing.assert_array_equal(modularity.next_ints(42, 7, 10), [1, 5, 6, 3, 5, 4, 1, 3, 6, 3])


def test_a_move_is_decided_in_plain_ieee_arithmetic():
    """The move rule rounds its product before subtracting it, as Seurat built for x86_64.

    Node 0 is alone, linked with weights 0.1 and 0.2 to two of the three nodes of cluster
    1; every node weighs 1 (``modularity_fxn=2``) and the resolution is 0.1. Joining
    gains (0.1 + 0.2) - 1 * 3 * 0.1. That is exactly 0 when 3 * 0.1 is rounded first, and
    2.8e-17 when one fused multiply-add computes it, as Seurat's arm64 build does. A gain
    of 0 leaves node 0 where it is.
    """
    rows, cols, weights = (0, 0, 1, 1, 2), (1, 2, 2, 3, 3), (0.1, 0.2, 1.0, 1.0, 1.0)
    A = sp.csc_matrix((weights + weights, (rows + cols, cols + rows)), shape=(4, 4))
    network = modularity.build_network(A, 2)
    cluster = np.array([0, 1, 1, 1])
    update, n_clusters = modularity._local_moving(
        network.node_weight, network.first_neighbor, network.neighbor, network.edge_weight,
        cluster, 2, 0.1, modularity.java_random(0))
    assert (0.1 + 0.2) - 1.0 * 3.0 * 0.1 == 0.0
    assert not update
    assert n_clusters == 2
    assert cluster[0] != cluster[1]


def test_only_the_lower_triangle_is_read():
    upper = sp.csc_matrix(np.triu(np.ones((4, 4)), k=1))
    with pytest.raises(ValueError, match="below its diagonal"):
        modularity.build_network(upper, 1)


def test_a_stored_zero_is_not_an_edge():
    """A stored zero would be kept by Seurat as an edge of weight 0, which can overrun
    one of its buffers; the network is built without it, and nothing else changes."""
    A = _asym().tocoo()
    zero_at = (A.row.max(), A.col.min())
    assert A.tocsr()[zero_at] == 0
    with_zero = sp.csc_matrix((np.append(A.data, 0.0), (np.append(A.row, zero_at[0]),
                                                        np.append(A.col, zero_at[1]))),
                              shape=A.shape)
    assert with_zero.nnz == A.nnz + 1
    for got, want in zip(modularity.build_network(with_zero, 1), modularity.build_network(A, 1)):
        np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# FindClusters, end to end
# ---------------------------------------------------------------------------

@pytest.fixture
def snn_object():
    """The SNN graph plus three cells that have no edges, on a truecell object."""
    snn = _snn()
    cells = [f"c{k}" for k in range(1, snn.shape[0] + 1)] + ["iso1", "iso2", "iso3"]
    rng = np.random.default_rng(0)
    obj = create_truecell_object(
        counts=sp.csr_matrix(rng.poisson(1.0, (20, len(cells))).astype(float)),
        feature_names=[f"g{k}" for k in range(20)], cell_names=cells)
    obj.graphs["RNA_snn"] = Graph(sp.block_diag((snn, sp.identity(3))).tocsc(), cells, "RNA")
    return obj


@pytest.mark.parametrize(
    "case", REFERENCE["find_clusters"],
    ids=lambda case: f"alg{case['algorithm']}-group_singletons={case['group_singletons']}")
def test_find_clusters_is_seurats(snn_object, case):
    """``FindClusters(graph, resolution = c(0.8, 1.6))``, singletons and all.

    The optimiser leaves each edgeless cell in a cluster of its own. ``GroupSingletons``
    then finds it tied across every cluster and places it where ``sample()`` says.
    """
    find_clusters(snn_object, resolution=case["resolution"], algorithm=case["algorithm"],
                  group_singletons=case["group_singletons"])
    for resolution, want in zip(case["resolution"], case["labels"]):
        got = snn_object.meta_data[f"RNA_snn_res.{resolution}"].astype(str).to_numpy()
        np.testing.assert_array_equal(got, want)


def _singleton_case(name):
    case = REFERENCE["group_singletons"][name]
    A = sp.csc_matrix((np.asarray(case["x"], dtype=float), (case["i"], case["j"])),
                      shape=(case["n"], case["n"]))
    ids = np.asarray(case["ids"])
    return case, _group_singletons(ids, A, True).tolist(), _group_singletons(ids, A, False).tolist()


def test_a_singleton_counts_towards_the_cluster_it_joined_for_the_next_one():
    """s1 joins cluster 0. s2 is best connected to cluster 0 only with s1 counted in it,
    because Seurat looks each cluster's cells up again for every singleton."""
    case, grouped, ungrouped = _singleton_case("recomputed")
    assert grouped == case["grouped"] == ["0", "0", "0", "1", "1", "1", "0"]
    assert ungrouped == case["ungrouped"]


def test_a_cell_with_no_edges_goes_where_seurats_sample_sends_it():
    """Both edgeless cells tie across all five clusters. Seurat draws among them in order
    of first appearance (3, 0, 4, 1, 2) with ``set.seed(1); sample()``, and gets 3, not
    the lowest label."""
    case, grouped, ungrouped = _singleton_case("isolated")
    assert grouped == case["grouped"]
    assert grouped[3] == grouped[-1] == "3"
    assert ungrouped == case["ungrouped"]


def test_a_tie_across_ten_clusters_takes_rs_ninth_candidate():
    """For eight or fewer tied clusters R's draw is the first to appear, which the
    cases above cannot tell from simply taking the first. With ten it is the ninth:
    cluster 6, where the first would be 7 and the lowest label 0."""
    case, grouped, ungrouped = _singleton_case("many")
    assert grouped == case["grouped"]
    assert grouped[-1] == "6"
    assert ungrouped == case["ungrouped"]


def test_an_exact_tie_goes_where_seurats_sample_sends_it():
    """Mean connectivity 0.4 to cluster 1 (two cells) and to cluster 0 (four cells)."""
    case, grouped, ungrouped = _singleton_case("tied")
    assert grouped == case["grouped"]
    assert grouped[-1] == "1"
    assert ungrouped == case["ungrouped"]


def test_the_tie_break_draw_is_rs():
    """``set.seed(1); sample.int(m, 1)`` for m = 1..300 and larger values, from R 4.6.1."""
    sample = REFERENCE["sample"]
    assert [_r_sample_index(m) + 1 for m in sample["m"]] == sample["draws"]


# ---------------------------------------------------------------------------
# The other arguments
# ---------------------------------------------------------------------------

def test_n_iterations_is_a_deprecated_name_for_n_iter(snn_object, monkeypatch):
    seen = []
    real = modularity.run_modularity_clustering

    def spy(network, resolution, algorithm, n_start, n_iter, *rest):
        seen.append(n_iter)
        return real(network, resolution, algorithm, n_start, n_iter, *rest)

    monkeypatch.setattr(modularity, "run_modularity_clustering", spy)
    with pytest.warns(FutureWarning, match="n_iter"):
        find_clusters(snn_object, n_iterations=3)
    assert seen == [3]


def test_leiden_turns_a_seed_of_0_into_1_as_seurat_does(snn_object, monkeypatch):
    """Seurat's RunLeiden warns and uses 1 for any seed of 0 or below, and FindClusters
    passes n.iter = 10."""
    pytest.importorskip("leidenalg")
    seen = []
    real = clustering._leiden_clustering

    def spy(graph, resolution, seed, n_iterations):
        seen.append((seed, n_iterations))
        return real(graph, resolution, seed, n_iterations)

    monkeypatch.setattr(clustering, "_leiden_clustering", spy)
    with pytest.warns(UserWarning, match="greater than 0"):
        find_clusters(snn_object, algorithm=4)
    assert seen == [(1, 10)]


def test_optimizer_igraph_is_one_igraph_pass(snn_object, monkeypatch):
    pytest.importorskip("igraph")
    called = []
    real = clustering._louvain_clustering

    def spy(graph, resolution, seed):
        called.append((resolution, seed))
        return real(graph, resolution, seed)

    monkeypatch.setattr(clustering, "_louvain_clustering", spy)
    monkeypatch.setattr(modularity, "run_modularity_clustering",
                        lambda *args: pytest.fail("the Seurat optimiser ran"))
    find_clusters(snn_object, optimizer="igraph", resolution=[0.5, 0.8])
    assert called == [(0.5, 0), (0.8, 0)]
