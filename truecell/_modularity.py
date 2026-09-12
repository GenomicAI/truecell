"""Seurat's modularity optimiser, ``RunModularityClusteringCpp``, in Python.

``FindClusters(algorithm = 1, 2 or 3)`` does not call igraph. It runs Seurat's C++
translation of Ludo Waltman and Nees Jan van Eck's ModularityOptimizer 1.3.0:

* 1, the Louvain algorithm (Blondel et al. 2008);
* 2, Louvain with multilevel refinement (Rotta & Noack 2011);
* 3, the smart local moving algorithm (Waltman & van Eck 2013).

Each of the ``n.start`` random starts begins from singleton clusters and repeats the
algorithm up to ``n.iter`` times, stopping early for 1 and 2 once a pass changes
nothing. The partition with the highest modularity is kept, and its clusters are
numbered by size. Every start draws from one ``java.util.Random`` stream, seeded
once by ``random.seed``.

This is a line-for-line translation, so that the same graph and seed give Seurat's
partition label for label. The order of floating-point operations and of random
draws is the C++'s, because both decide ties and the order nodes are visited in.
The recursion in the C++ is unrolled into loops here, which draws the random
numbers in the same order. The arithmetic is plain IEEE double, which is Seurat as
built for x86_64: an arm64 build fuses the multiply-add in the move rule and in
the modularity sum, and that flips moves whose gain is within one rounding of
zero, which only graphs with exactly tied weights produce.

References: Blondel, Guillaume, Lambiotte & Lefebvre (2008), J. Stat. Mech. P10008;
Rotta & Noack (2011), ACM J. Exp. Algorithmics 16, 2.3; Waltman & van Eck (2013),
Eur. Phys. J. B 86, 471.
"""
# Translated from Seurat 5.5.1 (src/ModularityOptimizer.h, src/ModularityOptimizer.cpp
# and src/RModularityOptimizer.cpp), which is distributed under this licence:
#
#   MIT License
#
#   Copyright (c) 2021 Seurat authors
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from numba import njit

_MULTIPLIER = np.uint64(0x5DEECE66D)
_INCREMENT = np.uint64(0xB)
_MASK_48 = np.uint64((1 << 48) - 1)
_SHIFT_17 = np.uint64(17)
_SHIFT_31 = np.uint64(31)


class Network(NamedTuple):
    """ModularityOptimizer's ``Network``: every undirected edge stored in both directions."""

    node_weight: np.ndarray
    first_neighbor: np.ndarray
    neighbor: np.ndarray
    edge_weight: np.ndarray
    self_links: float


# ---------------------------------------------------------------------------
# java.util.Random
# ---------------------------------------------------------------------------

def java_random(seed: int) -> np.ndarray:
    """The state of ``java.util.Random(seed)``, as a one-element array the kernels advance.

    Seurat takes the seed as a C ``int`` and widens it to 64 bits, so a negative seed
    is sign-extended before it is scrambled. Python's ``^`` on a negative ``int``
    does the same.
    """
    return np.array([(int(seed) ^ 0x5DEECE66D) & ((1 << 48) - 1)], dtype=np.uint64)


@njit(cache=True)
def _next_31(state):
    s = (state[0] * _MULTIPLIER + _INCREMENT) & _MASK_48
    state[0] = s
    return np.int64(s >> _SHIFT_17)


@njit(cache=True)
def _next_int(state, n):
    """``Random.nextInt(n)``, which rejects draws that would bias the remainder."""
    if (n & -n) == n:
        return np.int64((np.uint64(n) * np.uint64(_next_31(state))) >> _SHIFT_31)
    while True:
        bits = _next_31(state)
        val = bits % n
        # Java's test is `bits - val + (n - 1) < 0` on a 32-bit int, i.e. the sum
        # overflowed: `bits` fell in the last, incomplete block of n.
        if bits - val + (n - 1) < 2147483648:
            return val


@njit(cache=True)
def _random_permutation(n, state):
    permutation = np.arange(n)
    for i in range(n):
        j = _next_int(state, n)
        k = permutation[i]
        permutation[i] = permutation[j]
        permutation[j] = k
    return permutation


def next_ints(seed: int, n: int, size: int) -> np.ndarray:
    """``size`` successive ``nextInt(n)`` draws from ``java.util.Random(seed)``; for tests."""
    return _next_ints(java_random(seed), n, size)


@njit(cache=True)
def _next_ints(state, n, size):
    out = np.empty(size, np.int64)
    for i in range(size):
        out[i] = _next_int(state, n)
    return out


# ---------------------------------------------------------------------------
# Building the network
# ---------------------------------------------------------------------------

def build_network(graph: sp.spmatrix, modularity_fxn: int) -> Network:
    """``matrixToNetwork`` over the graph's strict lower triangle, as Seurat reads it.

    ``RunModularityClusteringCpp`` walks the matrix column by column and keeps entry
    (row, col) only when ``row > col``. Each kept entry becomes an undirected edge
    of that weight. The diagonal and the upper triangle are never read, so a graph
    that is not symmetric is clustered as its lower triangle mirrored. Every node's
    neighbours come out in ascending order.

    With ``modularity_fxn = 1`` a node's weight is the sum of its edge weights. With
    2, the alternative modularity, every node weighs 1.

    A stored zero is dropped before the walk. Seurat would keep it as an edge of
    weight 0, which changes no weight or decision below but can push the C++ past
    the end of a buffer. Dropping it keeps these kernels in bounds.
    """
    csc = sp.csc_matrix(graph, dtype=np.float64, copy=True)
    csc.eliminate_zeros()
    csc.sort_indices()
    n = max(csc.shape)
    if csc.shape[0] != csc.shape[1]:
        raise ValueError(f"The graph must be square; got {csc.shape[0]} x {csc.shape[1]}.")
    node_weight, first, neighbor, weight, n_input = _network(
        n, csc.indptr.astype(np.int64), csc.indices.astype(np.int64), csc.data,
        modularity_fxn == 2)
    if n_input == 0:
        raise ValueError("The graph has no edges below its diagonal, so there is nothing "
                         "to cluster.")
    return Network(node_weight, first, neighbor, weight, 0.0)


@njit(cache=True)
def _network(n, indptr, indices, data, unit_node_weights):
    n_neighbors = np.zeros(n, np.int64)
    n_input = 0
    for col in range(n):
        for p in range(indptr[col], indptr[col + 1]):
            row = indices[p]
            if col >= row:
                continue
            n_neighbors[col] += 1
            n_neighbors[row] += 1
            n_input += 1
    first = np.zeros(n + 1, np.int64)
    n_edges = 0
    for i in range(n):
        first[i] = n_edges
        n_edges += n_neighbors[i]
    first[n] = n_edges
    neighbor = np.empty(n_edges, np.int64)
    weight = np.empty(n_edges)
    n_neighbors[:] = 0
    for col in range(n):
        for p in range(indptr[col], indptr[col + 1]):
            row = indices[p]
            if col >= row:
                continue
            j = first[col] + n_neighbors[col]
            neighbor[j] = row
            weight[j] = data[p]
            n_neighbors[col] += 1
            j = first[row] + n_neighbors[row]
            neighbor[j] = col
            weight[j] = data[p]
            n_neighbors[row] += 1
    node_weight = np.ones(n)
    if not unit_node_weights:
        for i in range(n):
            total = 0.0
            for k in range(first[i], first[i + 1]):
                total += weight[k]
            node_weight[i] = total
    return node_weight, first, neighbor, weight, n_input


@njit(cache=True)
def _total_edge_weight(edge_weight):
    """``Network::getTotalEdgeWeight``: a running sum, each edge counted once."""
    total = 0.0
    for k in range(edge_weight.shape[0]):
        total += edge_weight[k]
    return total / 2.0


# ---------------------------------------------------------------------------
# VOSClusteringTechnique
# ---------------------------------------------------------------------------

@njit(cache=True)
def _local_moving(node_weight, first_neighbor, neighbor, edge_weight, cluster, n_clusters,
                  resolution, state):
    """``runLocalMovingAlgorithm``. Rewrites ``cluster``; returns (update, n_clusters)."""
    n = node_weight.shape[0]
    if n == 1:
        return False, n_clusters

    cluster_weight = np.zeros(n)
    n_nodes_per_cluster = np.zeros(n, np.int64)
    for i in range(n):
        cluster_weight[cluster[i]] += node_weight[i]
        n_nodes_per_cluster[cluster[i]] += 1

    n_unused = 0
    unused_cluster = np.zeros(n, np.int64)
    for i in range(n):
        if n_nodes_per_cluster[i] == 0:
            unused_cluster[n_unused] = i
            n_unused += 1

    node_permutation = _random_permutation(n, state)
    edge_weight_per_cluster = np.zeros(n)
    max_degree = 0
    for i in range(n):
        max_degree = max(max_degree, first_neighbor[i + 1] - first_neighbor[i])
    # The C++ sizes this n - 1; a node's degree bounds it as well and cannot overrun.
    neighboring_cluster = np.zeros(max(max_degree, 1), np.int64)

    update = False
    n_stable = 0
    i = 0
    while True:
        j = node_permutation[i]
        n_neighboring = 0
        for k in range(first_neighbor[j], first_neighbor[j + 1]):
            c = cluster[neighbor[k]]
            if edge_weight_per_cluster[c] == 0:
                neighboring_cluster[n_neighboring] = c
                n_neighboring += 1
            edge_weight_per_cluster[c] += edge_weight[k]

        current = cluster[j]
        cluster_weight[current] -= node_weight[j]
        n_nodes_per_cluster[current] -= 1
        if n_nodes_per_cluster[current] == 0:
            unused_cluster[n_unused] = current
            n_unused += 1

        best = -1
        max_quality = 0.0
        for k in range(n_neighboring):
            c = neighboring_cluster[k]
            quality = edge_weight_per_cluster[c] - node_weight[j] * cluster_weight[c] * resolution
            if quality > max_quality or (quality == max_quality and c < best):
                best = c
                max_quality = quality
            edge_weight_per_cluster[c] = 0
        if max_quality == 0:
            best = unused_cluster[n_unused - 1]
            n_unused -= 1

        cluster_weight[best] += node_weight[j]
        n_nodes_per_cluster[best] += 1
        if best == cluster[j]:
            n_stable += 1
        else:
            cluster[j] = best
            n_stable = 1
            update = True

        i = i + 1 if i < n - 1 else 0
        if n_stable >= n:
            break

    new_cluster = np.zeros(n, np.int64)
    n_clusters = 0
    for i in range(n):
        if n_nodes_per_cluster[i] > 0:
            new_cluster[i] = n_clusters
            n_clusters += 1
    for i in range(n):
        cluster[i] = new_cluster[cluster[i]]
    return update, n_clusters


@njit(cache=True)
def _nodes_per_cluster(cluster, n_clusters):
    """``Clustering::getNodesPerCluster``, flattened: nodes of cluster c are
    ``nodes[start[c]:start[c + 1]]``, in ascending order."""
    n = cluster.shape[0]
    start = np.zeros(n_clusters + 1, np.int64)
    for i in range(n):
        start[cluster[i] + 1] += 1
    for c in range(n_clusters):
        start[c + 1] += start[c]
    fill = start[:-1].copy()
    nodes = np.empty(n, np.int64)
    for i in range(n):
        c = cluster[i]
        nodes[fill[c]] = i
        fill[c] += 1
    return nodes, start


@njit(cache=True)
def _reduced_network(node_weight, first_neighbor, neighbor, edge_weight, self_links, cluster,
                     n_clusters):
    """``Network::createReducedNetwork``: one node per cluster."""
    nodes, start = _nodes_per_cluster(cluster, n_clusters)
    n_edges = neighbor.shape[0]
    reduced_node_weight = np.zeros(n_clusters)
    reduced_first = np.zeros(n_clusters + 1, np.int64)
    reduced_self_links = self_links
    neighbor_1 = np.empty(n_edges, np.int64)
    weight_1 = np.empty(n_edges)
    neighbor_2 = np.empty(n_edges + 1, np.int64)
    weight_2 = np.zeros(n_clusters)
    reduced_edges = 0
    for i in range(n_clusters):
        j = 0
        for p in range(start[i], start[i + 1]):
            v = nodes[p]
            reduced_node_weight[i] += node_weight[v]
            for m in range(first_neighbor[v], first_neighbor[v + 1]):
                c = cluster[neighbor[m]]
                if c != i:
                    if weight_2[c] == 0:
                        neighbor_2[j] = c
                        j += 1
                    weight_2[c] += edge_weight[m]
                else:
                    reduced_self_links += edge_weight[m]
        for k in range(j):
            neighbor_1[reduced_edges + k] = neighbor_2[k]
            weight_1[reduced_edges + k] = weight_2[neighbor_2[k]]
            weight_2[neighbor_2[k]] = 0
        reduced_edges += j
        reduced_first[i + 1] = reduced_edges
    return (reduced_node_weight, reduced_first, neighbor_1[:reduced_edges].copy(),
            weight_1[:reduced_edges].copy(), reduced_self_links)


@njit(cache=True)
def _refine_clusters(node_weight, first_neighbor, neighbor, edge_weight, cluster, n_clusters,
                     resolution, state):
    """The subnetwork step of ``runSmartLocalMovingAlgorithm``.

    Local moving runs inside each cluster's own subnetwork (singleton start, node
    weights kept from the full network), in cluster order. ``cluster`` is rewritten
    to the refined clusters. Returns how many refined clusters each original cluster
    split into, and their total.
    """
    n = node_weight.shape[0]
    nodes, start = _nodes_per_cluster(cluster, n_clusters)
    position = np.zeros(n, np.int64)
    refined = np.empty(n, np.int64)
    sizes = np.zeros(n_clusters, np.int64)
    offset = 0
    for i in range(n_clusters):
        m = start[i + 1] - start[i]
        sub_cluster = np.arange(m)
        sub_n_clusters = m
        if m > 1:
            n_sub_edges = 0
            for a in range(m):
                v = nodes[start[i] + a]
                position[v] = a
                n_sub_edges += first_neighbor[v + 1] - first_neighbor[v]
            sub_node_weight = np.empty(m)
            sub_first = np.zeros(m + 1, np.int64)
            sub_neighbor = np.empty(n_sub_edges, np.int64)
            sub_weight = np.empty(n_sub_edges)
            e = 0
            for a in range(m):
                v = nodes[start[i] + a]
                sub_node_weight[a] = node_weight[v]
                for k in range(first_neighbor[v], first_neighbor[v + 1]):
                    if cluster[neighbor[k]] == i:
                        sub_neighbor[e] = position[neighbor[k]]
                        sub_weight[e] = edge_weight[k]
                        e += 1
                sub_first[a + 1] = e
            _, sub_n_clusters = _local_moving(sub_node_weight, sub_first, sub_neighbor, sub_weight,
                                              sub_cluster, m, resolution, state)
        for a in range(m):
            refined[nodes[start[i] + a]] = offset + sub_cluster[a]
        offset += sub_n_clusters
        sizes[i] = sub_n_clusters
    cluster[:] = refined
    return sizes, offset


@njit(cache=True)
def _quality(node_weight, first_neighbor, neighbor, edge_weight, self_links, cluster, n_clusters,
             resolution):
    """``calcQualityFunction``: the modularity the restarts are ranked by."""
    n = node_weight.shape[0]
    quality = 0.0
    for i in range(n):
        j = cluster[i]
        for k in range(first_neighbor[i], first_neighbor[i + 1]):
            if cluster[neighbor[k]] == j:
                quality += edge_weight[k]
    quality += self_links
    cluster_weight = np.zeros(n_clusters)
    for i in range(n):
        cluster_weight[cluster[i]] += node_weight[i]
    for c in range(n_clusters):
        quality -= cluster_weight[c] * cluster_weight[c] * resolution
    quality /= 2 * _total_edge_weight(edge_weight) + self_links
    return quality


def _local(network: Network, cluster, n_clusters, resolution, state):
    return _local_moving(network.node_weight, network.first_neighbor, network.neighbor,
                         network.edge_weight, cluster, n_clusters, resolution, state)


def _reduce(network: Network, cluster, n_clusters) -> Network:
    return Network(*_reduced_network(network.node_weight, network.first_neighbor,
                                     network.neighbor, network.edge_weight, network.self_links,
                                     cluster, n_clusters))


def _louvain(network: Network, cluster, n_clusters, resolution, state, refine):
    """``runLouvainAlgorithm``, or ``...WithMultilevelRefinement`` when ``refine``.

    The C++ recurses into the reduced network. Here the descent collects the levels
    and the ascent merges each child's clustering into its parent, drawing random
    numbers in the order the recursion would.
    """
    levels = []
    while True:
        if network.node_weight.shape[0] == 1:
            update = False
            break
        update, n_clusters = _local(network, cluster, n_clusters, resolution, state)
        if n_clusters >= network.node_weight.shape[0]:
            break
        levels.append((network, cluster, n_clusters, update))
        network = _reduce(network, cluster, n_clusters)
        cluster = np.arange(n_clusters)

    child_cluster, child_n_clusters, child_update = cluster, n_clusters, update
    while levels:
        network, cluster, n_clusters, update = levels.pop()
        if child_update:
            update = True
            cluster[:] = child_cluster[cluster]
            n_clusters = child_n_clusters
            if refine:
                _, n_clusters = _local(network, cluster, n_clusters, resolution, state)
        child_cluster, child_n_clusters, child_update = cluster, n_clusters, update
    return child_cluster, child_n_clusters, child_update


def _smart_local_moving(network: Network, cluster, n_clusters, resolution, state):
    """``runSmartLocalMovingAlgorithm``, unrolled as :func:`_louvain` is."""
    levels = []
    while True:
        if network.node_weight.shape[0] == 1:
            update = False
            break
        update, n_clusters = _local(network, cluster, n_clusters, resolution, state)
        if n_clusters >= network.node_weight.shape[0]:
            break
        levels.append((cluster, update))
        sizes, n_refined = _refine_clusters(
            network.node_weight, network.first_neighbor, network.neighbor, network.edge_weight,
            cluster, n_clusters, resolution, state)
        network = _reduce(network, cluster, n_refined)
        # The reduced network starts from the clusters local moving found, not singletons.
        cluster = np.repeat(np.arange(n_clusters), sizes)

    child_cluster, child_n_clusters, child_update = cluster, n_clusters, update
    while levels:
        cluster, update = levels.pop()
        update = update or child_update
        cluster[:] = child_cluster[cluster]
        child_cluster, child_update = cluster, update
    return child_cluster, child_n_clusters, child_update


def order_clusters_by_size(cluster: np.ndarray) -> np.ndarray:
    """``Clustering::orderClustersByNNodes``: 0 is the largest, ties in cluster order."""
    counts = np.bincount(cluster)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(len(counts), np.int64)
    rank[order] = np.arange(len(counts))
    return rank[cluster]


def run_modularity_clustering(
    network: Network,
    resolution: float,
    algorithm: int,
    n_start: int,
    n_iter: int,
    random_seed: int,
    modularity_fxn: int,
) -> np.ndarray:
    """Seurat's ``RunModularityClusteringCpp`` on a built network: cluster ids by size.

    ``modularity_fxn`` must be the one the network was built with.
    """
    if modularity_fxn == 1:
        resolution = resolution / (2 * _total_edge_weight(network.edge_weight)
                                   + network.self_links)
    n = network.node_weight.shape[0]
    state = java_random(random_seed)
    best = None
    max_modularity = -np.inf
    for _ in range(n_start):
        cluster = np.arange(n)
        n_clusters = n
        update = True
        iteration = 0
        while True:
            if algorithm == 1:
                cluster, n_clusters, update = _louvain(network, cluster, n_clusters, resolution,
                                                       state, refine=False)
            elif algorithm == 2:
                cluster, n_clusters, update = _louvain(network, cluster, n_clusters, resolution,
                                                       state, refine=True)
            else:
                # Seurat does not read this one's return value, so every iteration runs.
                cluster, n_clusters, _ = _smart_local_moving(network, cluster, n_clusters,
                                                             resolution, state)
            iteration += 1
            modularity = _quality(network.node_weight, network.first_neighbor, network.neighbor,
                                  network.edge_weight, network.self_links, cluster, n_clusters,
                                  resolution)
            if not (iteration < n_iter and update):
                break
        if modularity > max_modularity:
            best = cluster
            max_modularity = modularity
    if best is None:
        raise RuntimeError("Clustering failed: no random start produced a finite modularity.")
    return order_clusters_by_size(best)
