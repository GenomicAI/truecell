# Differential expression

`find_markers` implements all nine of Seurat's tests: `wilcox` (tie-corrected),
`t`, `bimod`, `LR`, `negbinom`, `poisson`, `roc`, `mast` and `deseq2`. On PBMC 3k
the eight that return a p-value reproduce Seurat's top 50 genes exactly, and
`roc` its AUCs to Seurat's three-decimal rounding — see
[the DE vignette](../tutorials/de_vignette.md). `deseq2` tests every cell as a
replicate, as Seurat's `DESeq2DETest` does; `sample_col` sums each sample's cells
first and makes it a pseudobulk test.

Two numbers to know before reading a result table:

- **`avg_log2FC` carries Seurat's pseudocount on the group *sum*, not the group
  mean.** Getting that backwards shifts every fold change and also changes which
  genes clear `logfc_threshold`, so it silently changes the returned gene set,
  not just a column.
- **`pct.1` and `pct.2` are rounded to three decimals** before `min_pct` filters
  on them, as Seurat's `FoldChange` rounds them, and by R's rule rather than
  NumPy's. A gene detected in 19 of 2,000 cells (0.0095) therefore passes
  `min_pct=0.01` in both tools, and the two columns match Seurat's exactly.

## Per-cluster and per-pair tests

::: truecell.markers.find_markers

::: truecell.markers.find_all_markers

::: truecell.markers.find_conserved_markers

## Group summaries

`AggregateExpression` **sums raw counts** and is what pseudobulk differential
expression wants. `AverageExpression` **means the back-transformed values** and
is what a per-group expression summary wants. They are different functions, not
two scalings of one — see each docstring.

::: truecell.aggregate.aggregate_expression

::: truecell.aggregate.average_expression
