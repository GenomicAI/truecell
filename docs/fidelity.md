# Fidelity

A port is only worth as much as its evidence. This page is how `truecell` is
checked against R Seurat, what the checking has found, and where the two tools
genuinely disagree.

## How the checking works

Every [tutorial](tutorials/README.md) is three files, not one:

| File | What it does |
|---|---|
| `tutorials/<name>_tutorial.py` | Runs the analysis in Python and writes a numeric handoff |
| `tutorials/<name>_verify.R` | Runs the same analysis under real Seurat and writes its own |
| `tutorials/<name>_vignette.md` | The write-up, with both sides' code and the comparison |

```bash
python tutorials/pbmc3k_de_tutorial.py            # writes the handoff
Rscript tutorials/pbmc3k_de_verify.R              # reads it, writes R's answers
python tutorials/pbmc3k_de_tutorial.py --report   # compares; exits non-zero if it should
```

**Order matters, and the tools now enforce it.** Python runs first because it
writes the handoff the R script reads. The references were taken on
**Seurat 5.5.1**.

### Two kinds of comparison, deliberately

Some tutorials pin nothing: both sides start from the same 10x bytes and run
their own pipeline end to end. That tests what a user actually experiences — the
same cells surviving QC, the same variable genes, the same clusters. PBMC 3k is
one of these.

Others pin the cells and features and hand them to R explicitly, so a difference
in an earlier stage cannot pose as a difference in the one under test. The DE
suite is the clearest case: all nine tests run on **one shared cluster
assignment**, because otherwise a clustering difference would surface as a DE
difference and be attributed to the wrong function.

### Anchors, not vibes

Each comparison declares specific quantities. An *anchor* is one named number
that must match, at a stated tolerance or exactly. "91 of 91 anchors match, no
tolerance" is a claim you can check line by line; "the object model works" is
not.

## What the checking has caught

The tutorials are the debugging apparatus, not a showcase. Each vignette names
the defects its own comparison found:

| Tutorial | Found |
|---|---|
| [Anchor internals](tutorials/anchors_vignette.md) | **18** — CCA standardizing instead of L2-normalizing, randomized SVD drifting reciprocal-PCA's trailing PCs, `integrate_layers` silently running v4's algorithm behind the v5 name, a symmetrized KNN graph, a missing SNN diagonal |
| [The object model](tutorials/objects_vignette.md) | **11** — a split/join round trip that silently misordered columns, `FetchData` returning sparse objects instead of numbers, an inert command log |
| [Batch integration](tutorials/integration_vignette.md) | **8** — a crash on unequal batch sizes, and a 4× under-integration: RPCA's batch mixing 0.222 → 0.867 → 0.991, on truecell's clusters before `find_clusters` ran Seurat's optimiser (0.917 now, as in Seurat) |
| [Out of core](tutorials/lazy_vignette.md) | **7** — five functions that densified the whole store, so going on disk *raised* peak memory 4.6× |
| [Spatial statistics](tutorials/svf_vignette.md) | **3** — Moran's I on a kNN graph instead of R's inverse-square weights, centroids with no radius, unclosed polygons |
| [Dim-reduction extras](tutorials/dimreduc_vignette.md) | **2** — a too-tight JackStraw null, and the wrong aggregation test |
| [Leverage sketching](tutorials/sketch_vignette.md) | **2** — full-rank leverage, anchor-based label transfer |
| [The DE test suite](tutorials/de_vignette.md) | **2** — Seurat's pseudocount applied to the group mean instead of the sum, which also changed *which genes* `logfc_threshold` returned |
| [Visium](tutorials/visium_vignette.md) | **1 in truecell, and 1 in Seurat** — the first tutorial where R is the one that's wrong |

Two lessons worth stating outright, because both cost real time:

**A documented caveat is where a defect hides.** Twice, a genuine bug sat behind
a comment explaining it as an expected language difference — the SCTransform
model was one, JackStraw's null another. A "known difference" nobody has measured
is a hypothesis, not a finding.

**A fix can make the headline number worse and still be the fix.** Sketching's
`project_data` scored *above* Seurat while it was broken. When a divergence
flatters the port, that is a reason to look harder, not to keep it.

## What actually differs { #what-actually-differs }

Real, understood, and not going away:

**Clusters differ only where the graphs do.** `find_clusters` runs Seurat's own
modularity optimiser, so the same graph gives the same partition: every one of 12
runs on PBMC 3k's SNN graph, and Seurat's 16 clusters on ifnb's RPCA graph, cell
for cell. End to end each tool builds its own graph, and on PBMC 3k the two
partitions agree at ARI 0.928, nine clusters on both sides. Where edge weights tie
exactly, Seurat's own partition depends on the processor: built for arm64, its C++
fuses a multiply-add that flips near-tied moves. truecell follows the x86_64 build,
as it does for `clara`.

**Variable-feature selection jitters at the boundary.** 1,998 of 2,000 genes
shared on PBMC 3k. The genes that swap sit at ranks 1916–2016, where
standardized variances differ in the third decimal. Harmless in itself — but it
changes which genes a later stage sees, which is why references have to be
regenerated together rather than piecemeal.

**Anything with an RNG differs by its RNG, and only by that.** `add_module_score`
draws control genes at random: per-cell phase calls come out 95.9 % concordant,
the rate two NumPy seeds agree with each other, and the continuous scores
correlate at Pearson ≥ 0.997. JackStraw permutes.
Differences here get proved distribution-against-distribution over matched seeds,
never from a single pair — single-run pairs were actively misleading on the
sketch composition.

**R's `clara` is architecture-dependent.** It gives different answers on arm64
and x86_64. The port targets IEEE/x86_64 semantics on purpose rather than
emulating whichever one a given laptop produces.

**Seurat's default neighbour search is approximate.** `annoy` against an exact
search is a difference in the *reference*, not in either implementation. The
verify scripts pass `nn.method = "rann"`; skipping that once cost one script a
false negative of 182 SNN edges.

## What is deliberately not compared, and why { #not-compared }

**UMAP coordinates.** `run_umap` is validated as an *implementation
correspondence* — same algorithm, same parameters, same input embedding — and its
output coordinates are deliberately excluded from every agreement metric in this
project. That is a statement about UMAP, not about the port.

A UMAP embedding is not identified up to anything stronger than its topology. The
optimisation is stochastic and non-convex, so the layout is fixed only up to
rotation, reflection and the particular local optimum a run lands in; R Seurat
drives [`uwot`](https://github.com/jlmelville/uwot) and truecell drives
[`umap-learn`](https://umap-learn.readthedocs.io), and the two differ in
initialisation strategy, neighbour-graph construction and optimisation detail.
Two runs of the *same* library on the same data with different seeds already
disagree on coordinates. A coordinate-level agreement number would therefore
measure the seed and the library, and a *high* one would be the surprising
result.

What carries the weight instead is everything either side of it: the neighbour
graph the embedding is computed from is compared directly, and the cluster
assignments plotted on it are scored by ARI. If those agree, the picture agrees
in the only sense a UMAP picture means anything — which clusters exist and which
cells are in them, not where on the page they landed.

The same reasoning applies to any figure-level comparison: plots are checked for
*what they encode*, never for pixel agreement.

## Numbers that move are declared as bands { #bands }

A comparison that legitimately varies used to carry a prose caveat — "expect
around 22 of 50 here" — and prose does not fail a build. A regression landing
inside the expected spread read as normal variation.

`tutorials/bands.py` replaces those caveats with `Band` objects carrying a range
**and the reason for it**. `--report` prints a verdict per band and exits
non-zero outside one. Two rules make them worth having:

- **A band on a number that moves came from a sweep, not from one run.** The
  JackStraw band is 60 seeds.
- **A missing measurement fails.** `Band.holds(nan)` is `False`, because a
  measurement that quietly vanished is exactly how a stale reference goes
  unnoticed.

| Band | Range | Why it is where it is |
|---|---|---|
| JackStraw PC cutoff | \|truecell − R\| ≤ 2 | R's `JackRandom` seeds each replicate from its loop index, so R is deterministic at 13. truecell seeds from its `seed` argument and keeps 12/13/14/15 for 2/28/11/19 of 60 seeds — mode 13, which is R's answer. |
| The eight DE p-value tests | exactly 50 | One dropped gene is a regression. All eight gave 50 of 50 both before and after `find_clusters` began running Seurat's optimiser, which moved cells between the two clusters. At `deseq2`'s cut the 50th and 51st genes sit 5.0 decades apart in truecell and 4.9 in Seurat, and no gene within three ranks of it differs by more than 0.05. |
| `roc` max ∆AUC | ≤ 5e-4 | Half a unit in Seurat's third decimal. Measured 4.9986e-4 — bands are inclusive by design, and this one sits on its boundary. |

`max |Δlog2FC|` is one band over every test, `deseq2` included: all of them
report Seurat's fold change on the same matrix, so all of them agree to 1.8e-15,
one unit in the last place of a double, read against R's hex-float tables.

## The reference has to be the one the handoff asked for

Three of the comparison paths would silently accept an out-of-date R reference,
and one was actively doing so: a Python run from 25 July compared against R files
from 19 July, taken on a *different* cluster assignment, printing a full parity
table that showed `wilcox` at 48 of 50 with p-value Spearman 0.907. Against a
matching R run it is 50 of 50 at 1.000000. The regression was in the reference.

The guards now in `tutorials/bands.py`:

- **DE**: `pct.1` and `pct.2` must agree to 5e-4. They are counts of detected
  cells per group with no statistics in the way, which is what makes them the
  discriminator — they had differed for 12,491 of 13,712 genes. Seurat rounds
  them to three decimals and truecell now rounds them the same way, so a current
  run agrees exactly; the 5e-4 allows for a table written before it did.
- **Dim-reduction**: exact cell-set and feature-set equality. The old code
  reindexed R's frame onto Python's cells, which filled absent barcodes with NaN
  and printed `nan` correlations without complaint.
- **`pytest` no longer corrupts the handoff.** `prep()` wrote the HVG and cell
  lists unconditionally, so running the test suite replaced 2,000 real variable
  features with 100 synthetic genes named `GENE171` — which the R verify script
  then read.

## The annotations that only exist for readers { #the-annotations-that-only-exist-for-readers }

Three groups of annotations in this package name symbols that do not exist at
runtime. All are guarded by `if TYPE_CHECKING:`, and all are deliberate:

- `matplotlib.figure.Figure`, on all 17 plotting functions, so matplotlib stays
  an optional dependency;
- `Neighbor` in `graph.py`, because `Graph` and `Neighbor` convert into each
  other and a module-scope import would be a genuine cycle;
- `Truecell` in `compat/anndata.py`, for the same reason.

**`typing.get_type_hints()` raises `NameError` on all 19 of them**, and that was
recorded as a blocker for this documentation site. It turned out not to be one.
mkdocstrings reads annotations *statically*, through griffe, without importing
the module — so the signatures render and the types cross-link precisely
*because* the deferred imports are there. Runtime resolution is a separate
question that nothing here needs an answer to.

## Reproducing any of it

```bash
git clone https://github.com/GenomicAI/truecell.git
cd truecell && uv venv && uv pip install -e ".[all]"

python tutorials/pbmc3k_de_tutorial.py
Rscript tutorials/pbmc3k_de_verify.R
python tutorials/pbmc3k_de_tutorial.py --report
```

Every vignette's header names the datasets it needs and the R packages beyond
Seurat. Data downloads on first run to `~/.truecell_data/`; the full set is about
770 MB.
