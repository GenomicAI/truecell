# Graphs and clustering

`find_neighbors` builds both graphs Seurat builds — the directed KNN graph and
the shared-nearest-neighbour graph on top of it — and `find_clusters` runs
community detection over the SNN.

For Louvain, Louvain with multilevel refinement and smart local moving,
`find_clusters` runs Seurat's own modularity optimiser: a translation of the C++
behind `FindClusters`, with the same restarts and the same random-number stream.
Given the same graph it returns Seurat's partition, label for label. Leiden runs
through leidenalg.

Four details here were wrong before the integration tutorial went looking, and
each mattered downstream: the KNN graph is stored directed rather than
symmetrized, the SNN keeps its diagonal and is computed in float64, `run_umap`
zeroes the diagonal when handed a graph, and singletons are folded into their
nearest community the way `GroupSingletons` does. They are noted in the
docstrings because each one changes cluster assignments, not just internals.

## Neighbour graphs

::: truecell.neighbors.find_neighbors

::: truecell.multimodal.find_multi_modal_neighbors

## Community detection

::: truecell.clustering.find_clusters
