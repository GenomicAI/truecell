---
name: truecell-differential-expression
description: Use when finding marker genes or running differential expression in truecell — find_markers, find_all_markers, find_conserved_markers, aggregate_expression, choosing among the nine test_use options (wilcox, t, bimod, LR, negbinom, poisson, mast, deseq2, roc), pseudobulk DE with sample_col, or interpreting the output columns and fold changes.
---

# Differential expression in truecell

Load the `truecell` skill first. All four functions here **return** a DataFrame;
none mutate the object.

## The three entry points

```python
import truecell

# One group against another (None = against all other cells)
markers = truecell.find_markers(obj, ident_1="1")
markers = truecell.find_markers(obj, ident_1="1", ident_2="3")

# Every cluster against all others
all_markers = truecell.find_all_markers(obj, only_pos=True, min_pct=0.25, logfc_threshold=0.25)

# Up in cluster 1 in *every* condition (per-condition test, Fisher-combined p)
conserved = truecell.find_conserved_markers(obj, ident_1="1", grouping_var="condition")
```

`ident_1` / `ident_2` match the **active identity** (`obj.idents`), which
`find_clusters` sets to the cluster labels. To test something else, set the
identity first:

```python
obj.meta_data["seurat_clusters_orig"] = obj.idents      # keep the clusters
obj = obj.rename_idents(...)                            # or set from a metadata column
```

## Choosing `test_use`

Nine tests, all of them Seurat's. The eight that return a p-value were verified
to reproduce Seurat's **top 50 genes exactly** on a shared cluster assignment,
with `avg_log2FC` agreeing to 7.1e-15.

| `test_use` | What it is | Reach for it when |
|---|---|---|
| `"wilcox"` | Tie-corrected rank-sum. **Default.** | Almost always. Fast, non-parametric, the field's convention. |
| `"t"` | Student's t on log-normalized data | A parametric cross-check on `wilcox`. |
| `"bimod"` | McDavid 2013 bimodal likelihood-ratio test | Genes that are bimodal rather than shifted — on/off rather than up/down. |
| `"LR"` | Logistic-regression LRT | You need covariates: pass `latent_vars=["percent.mt"]`. |
| `"negbinom"` | Negative-binomial GLM Wald test **on counts** | UMI counts directly, no normalization assumption. Slow. |
| `"poisson"` | Poisson GLM Wald test **on counts** | Counts, when speed matters more than calibration. Anti-conservative on overdispersed UMIs — prefer `negbinom`. |
| `"mast"` | MAST two-part hurdle LRT | The hurdle model — detection and magnitude tested jointly. Supports `latent_vars`. |
| `"deseq2"` | DESeq2 as Seurat runs it, on counts | Reproducing a Seurat DESeq2 result (every cell a replicate), or sample-level inference with `sample_col=`. Needs `pip install truecell[deseq2]`. |
| `"roc"` | AUC classifier power | Ranking markers by how well they separate, with no p-value at all. |

Practical default: `wilcox` for discovery, `deseq2` with `sample_col` for
anything where a claim depends on replicate-level significance.

## Before DE on a merged SCT assay

If the object carries **more than one** SCT model — several objects
SCTransformed separately, then merged — run `prep_sct_find_markers(obj)` first:

```python
merged = a.merge(b, add_cell_ids=["ctrl", "stim"])
truecell.prep_sct_find_markers(merged)          # once, before any find_markers
truecell.find_markers(merged, "0", "1", assay="SCT")
```

SCTransform corrects each object's counts to *its own* median sequencing depth,
so without this a fold change across the merge partly measures how deeply each
batch was sequenced. It is a no-op on a single-model object, so it is safe to
call unconditionally.

## Output columns

For `wilcox` / `t` / `bimod` / `LR` / `negbinom` / `poisson` / `mast` / `deseq2`, sorted by `p_val`:

| Column | Meaning |
|---|---|
| `p_val` | Raw p-value |
| `avg_log2FC` | log2 fold change, group 1 over group 2 |
| `pct.1` | Fraction of group-1 cells detecting the gene |
| `pct.2` | Fraction of group-2 cells detecting the gene |
| `p_val_adj` | Bonferroni-corrected over all tested genes |

`roc` returns a different frame — `myAUC`, `avg_diff`, `power`, `avg_log2FC`,
`pct.1`, `pct.2`, sorted by `power`, **with no p-value**. Matching Seurat. Code
that assumes `p_val` exists breaks on `roc`.

`find_all_markers` adds `cluster` and `gene` columns and filters at
`return_thresh=0.01` on `p_val` (`power` for `roc`).

### `avg_log2FC` is computed Seurat's way

Seurat's pseudocount goes on the **summed** expression per group, not on the
group mean. truecell once had it on the mean; the fix changed not only the values
but *which genes passed `logfc_threshold`*. If you are reimplementing the
comparison, match the summation, not just the formula's shape.

## Filters

```python
truecell.find_markers(obj, ident_1="1", min_pct=0.01, logfc_threshold=0.1, only_pos=False)
```

These are Seurat 5's defaults. Seurat 4 used `min_pct=0.1` and
`logfc_threshold=0.25`; pass those explicitly to reproduce an analysis run on it.

- `min_pct=0.01` — the gene must be detected in ≥1 % of cells in **either** group.
  The fraction is rounded to three decimals first, exactly as Seurat rounds it.
- `logfc_threshold=0.1` — |log2FC| floor, applied **before** testing. It is a
  speed filter with statistical consequences: raise it and you lose genuinely
  significant small-effect genes.
- `only_pos=True` — up in group 1 only. Standard for cluster markers.
- `features=[...]` — restrict to a gene list, e.g. a pathway.
- `max_cells_per_ident=N` — downsample each group; use on huge objects and
  record that you did, since it changes the p-values.

## Covariates

```python
truecell.find_markers(obj, ident_1="stim", ident_2="ctrl",
                    test_use="LR", latent_vars=["percent.mt", "nCount_RNA"])
```

`latent_vars` is honoured by `LR`, `negbinom`, `poisson` and `mast` only —
the same four as Seurat's `DEmethods_latent()`.

**A deliberate MAST difference:** Seurat's `MASTDETest` fits `~ condition` alone
and adds **no** cellular detection rate term. Leaving `latent_vars` empty is what
matches Seurat. Adding CDR is the MAST paper's advice and a departure from
Seurat — do it knowingly, and say so in the write-up.

## Pseudobulk

Two things, often confused.

**Aggregate the counts** (for export, or for an external DE tool):

```python
pb = truecell.aggregate_expression(obj, group_by=["cell_type", "donor"], layer="counts")
pb_obj = truecell.aggregate_expression(obj, group_by="cell_type", return_object=True)
```

Sums counts per group. `return_object=True` gives a Truecell object with one cell
per group instead of a matrix.

`average_expression` (Seurat's `AverageExpression`) is the **mean** counterpart —
same grouping arguments, but the per-group *mean* of the `data` layer rather
than the sum of `counts`. Use `aggregate_expression` for anything feeding a
count-based test such as DESeq2, and `average_expression` for a per-group
expression summary you intend to read or plot:

```python
avg = truecell.average_expression(obj, group_by="cell_type")     # genes x groups
```

**Test at the sample level** with DESeq2:

```python
# obj.idents must hold the two conditions being compared
de = truecell.find_markers(obj, ident_1="stim", ident_2="ctrl",
                         test_use="deseq2", sample_col="donor")
```

`sample_col` names the metadata column identifying the biological replicates.
Counts are summed per (group, sample), and those profiles go through the same
DESeq2 test. Every other test ignores it.

**Without `sample_col`, `deseq2` is Seurat's per-cell test.** `DESeq2DETest` gives
DESeq2 one column per cell, and so does truecell. On PBMC 3k it calls the same
726 genes as Seurat at `p_val_adj < 0.05`, with the same top 50. Two things
follow from DESeq2 itself:

- Cells are not independent replicates, so per-cell p-values are
  anti-conservative. Use it to reproduce a Seurat analysis, and `sample_col` for
  a claim about conditions.
- DESeq2's size factors need at least one gene with no zero in any cell. When no
  gene qualifies, `find_markers` raises, as `estimateSizeFactors` stops in R.
  Aggregate with `sample_col`, or compare fewer cells.

`avg_log2FC` is Seurat's fold change on the data layer for every test, `deseq2`
included, not DESeq2's own `log2FoldChange`.

## Conserved markers

```python
conserved = truecell.find_conserved_markers(obj, ident_1="1", grouping_var="condition")
```

Runs the test separately within each level of `grouping_var` and combines the
p-values with Fisher's method. Use it when a marker must hold in every
condition/batch, not just on average across them.

## Interpreting results honestly

- `p_val_adj` is Bonferroni over genes tested **in that call**. Changing
  `logfc_threshold` or `features` changes the correction. Two runs' adjusted
  p-values are not comparable unless the tested gene set was the same.
- Per-cell tests treat cells as independent replicates. They are not. Where the
  question is "does this differ between conditions", `deseq2` with `sample_col`
  is the defensible answer and `wilcox` is the exploratory one.
- A marker table is downstream of a clustering. Report which clustering — and
  when comparing tools, hand both sides the **same** cell assignment, or a
  clustering difference will surface as a DE difference and be blamed on the
  wrong function.

## Reference

[The DE test-suite vignette](https://genomicai.github.io/truecell/tutorials/de_vignette/)
runs all nine tests against `FindMarkers` on one shared cluster assignment.
