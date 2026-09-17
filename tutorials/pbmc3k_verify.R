#!/usr/bin/env Rscript
# R Seurat reference for the PBMC 3k guided-clustering tutorial
# (pbmc3k_tutorial.md).
#
# Runs the standard PBMC 3k workflow (mirrors pbmc3k_tutorial.py) and writes,
# into tutorials/figures/:
#   * r_12_ridge_plot.png   the one figure the published Seurat vignette omits
#   * r_cell_meta.csv       per-cell QC, cluster, cell type and PC_1..PC_10
#   * r_hvg.csv             per-gene VST statistics + the selected 2,000
#   * r_markers.csv         FindAllMarkers table (cluster, gene, stats)
#   * r_anchors.json        scalars: counts, graph nnz, PC stdevs, data sum
#
# Nothing here is pinned to the Python run. Both sides start from the same 10x
# bytes and run their own pipeline end to end, so the comparison tests what the
# tutorial actually claims: that the two arrive at the same cells, the same
# variable genes, the same clusters and the same markers on their own.
# (`pbmc3k_dimreduc_verify.R` is the opposite experiment — it *does* pin
# Python's cells and features, to isolate the post-PCA machinery.)
#
# Data: the same cache the Python tutorial downloads to. Run
#   python tutorials/pbmc3k_tutorial.py     # downloads ~24 MB first
# then
#   Rscript tutorials/pbmc3k_verify.R
#   python tutorials/pbmc3k_tutorial.py --report
# Override the data folder with the PBMC3K_DATA environment variable.
#
# Needs: Seurat, ggplot2, ggridges, jsonlite.
suppressPackageStartupMessages({
  library(Seurat); library(ggplot2); library(jsonlite)
})
set.seed(0)

.args <- commandArgs(trailingOnly = FALSE)
.script <- sub("^--file=", "", .args[grep("^--file=", .args)])
HERE <- if (length(.script)) dirname(normalizePath(.script)) else getwd()
FIG  <- file.path(HERE, "figures")
DATA <- Sys.getenv("PBMC3K_DATA",
                   path.expand("~/.truecell_data/pbmc3k/filtered_gene_bc_matrices/hg19"))
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(file.path(DATA, "matrix.mtx")))
  stop("PBMC 3k data not found at ", DATA,
       "\nRun `python tutorials/pbmc3k_tutorial.py` first.")

# ---- standard workflow (mirrors pbmc3k_tutorial.py) -------------------------
raw <- CreateSeuratObject(Read10X(DATA), project = "pbmc3k",
                          min.cells = 3, min.features = 200)
raw[["percent.mt"]] <- PercentageFeatureSet(raw, pattern = "^MT-")
n_genes_raw <- nrow(raw); n_cells_raw <- ncol(raw)
pbmc <- subset(raw, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize",
                      scale.factor = 10000, verbose = FALSE)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
pbmc <- ScaleData(pbmc, features = rownames(pbmc), verbose = FALSE)
pbmc <- RunPCA(pbmc, npcs = 50, verbose = FALSE)
# nn.method = "rann" gives exact neighbours. Seurat's default is "annoy", which
# is approximate, while truecell's neighbour search is exact — leaving the default
# in place would compare two different neighbour tables and report a difference
# that belongs to annoy rather than to either implementation. The same trap cost
# `pbmc3k_objects_verify.R` a false negative of 182 SNN edges.
pbmc <- FindNeighbors(pbmc, dims = 1:10, k.param = 20, nn.method = "rann", verbose = FALSE)
# The vector form, matching the Python side's RESOLUTION_SWEEP. Each resolution
# lands in its own `RNA_snn_res.<r>` column, and Seurat leaves the object on the
# **last** one given — last as given, not largest — so 0.5 stays here at the end
# and every step below sees the partition it always saw.
pbmc <- FindClusters(pbmc, resolution = c(0.4, 0.8, 1.2, 0.5), algorithm = 1,
                     verbose = FALSE)
res_cols <- c("RNA_snn_res.0.4", "RNA_snn_res.0.8",
              "RNA_snn_res.1.2", "RNA_snn_res.0.5")
write.csv(
  data.frame(cell = colnames(pbmc),
             setNames(lapply(res_cols, function(c) as.character(pbmc[[c]][, 1])),
                      res_cols),
             check.names = FALSE, stringsAsFactors = FALSE),
  file.path(FIG, "r_resolutions.csv"), row.names = FALSE)
cat("Wrote r_resolutions.csv (", paste(res_cols, collapse = ", "), ")\n")
pbmc <- RunUMAP(pbmc, dims = 1:10, verbose = FALSE)
cat(sprintf("PBMC 3k: %d cells -> %d after QC, %d clusters\n",
            n_cells_raw, ncol(pbmc), length(levels(pbmc))))

pbmc$seurat_clusters <- Idents(pbmc)
all_markers <- FindAllMarkers(pbmc, only.pos = TRUE, min.pct = 0.25,
                              logfc.threshold = 0.25, verbose = FALSE)

# ---- cell-type annotation (_assign_cell_types ported from the Python side) ---
# The tutorial's RenameIdents() step. Each cluster scores a point for each
# canonical panel gene among its top-50 markers, each cell type names at most one
# cluster, and the assignment with the highest total wins. Among assignments with
# the same total, the earliest cluster takes the earliest panel that still allows
# it — so the loop order over clusters and panels is part of the definition.
# Iterate clusters numerically (0, 1, ..., 10, 11) and keep the panels in the
# Python helper's order.
MARKERS_REF <- list(
  "Naive CD4 T"  = c("IL7R","CCR7"),   "CD14+ Mono"   = c("CD14","LYZ"),
  "Memory CD4 T" = c("IL7R","S100A4"), "B"            = c("MS4A1"),
  "CD8 T"        = c("CD8A"),          "FCGR3A+ Mono" = c("FCGR3A","MS4A7"),
  "NK"           = c("GNLY","NKG7"),   "DC"           = c("FCER1A","CST3"),
  "Platelet"     = c("PPBP")
)
assign_cell_types <- function(markers, obj) {
  clusters <- as.character(sort(unique(as.integer(as.character(Idents(obj))))))
  types <- names(MARKERS_REF)
  n <- length(clusters); k <- length(types)
  score <- matrix(0, n, k)
  for (i in seq_len(n)) {
    df <- markers[as.character(markers$cluster) == clusters[i], ]
    top50 <- head(df$gene, 50)
    for (j in seq_len(k)) score[i, j] <- sum(MARKERS_REF[[types[j]]] %in% top50)
  }
  # best(i, used): the highest total clusters i..n can still reach when the
  # types in the bitmask `used` are taken. Nine panels make 512 masks.
  memo <- matrix(NA_real_, n + 1, 2^k)
  best <- function(i, used) {
    if (i > n) return(0)
    if (is.na(memo[i, used + 1])) {
      total <- best(i + 1, used)  # cluster i left "Unknown"
      for (j in seq_len(k)) {
        bit <- bitwShiftL(1L, j - 1L)
        if (score[i, j] > 0 && bitwAnd(used, bit) == 0)
          total <- max(total, score[i, j] + best(i + 1, bitwOr(used, bit)))
      }
      memo[i, used + 1] <<- total
    }
    memo[i, used + 1]
  }
  assignment <- setNames(rep("Unknown", n), clusters); used <- 0L
  for (i in seq_len(n)) {
    for (j in seq_len(k)) {
      bit <- bitwShiftL(1L, j - 1L)
      if (score[i, j] > 0 && bitwAnd(used, bit) == 0 &&
          score[i, j] + best(i + 1, bitwOr(used, bit)) == best(i, used)) {
        assignment[i] <- types[j]; used <- bitwOr(used, bit)
        break
      }
    }
  }
  assignment
}
anno <- assign_cell_types(all_markers, pbmc)
cat("Cluster -> cell type:\n"); print(anno)
pbmc$celltype <- unname(anno[as.character(pbmc$seurat_clusters)])

# ======================= NUMERIC HANDOFF ====================================
# The figures are compared by eye; these files are not.
#
# Everything per cell is keyed by *barcode* and everything per gene by *symbol*,
# so the report can ask each question on a shared key rather than on two
# independently-ordered tables. That is what lets it separate "the clusters are
# numbered differently" from "the clusters contain different cells".
emb <- Embeddings(pbmc, "pca")[, 1:10]
colnames(emb) <- paste0("PC_", 1:10)
write.csv(cbind(data.frame(
  cell         = colnames(pbmc),
  nCount_RNA   = as.numeric(pbmc$nCount_RNA),
  nFeature_RNA = as.numeric(pbmc$nFeature_RNA),
  percent.mt   = as.numeric(pbmc$percent.mt),
  cluster      = as.character(pbmc$seurat_clusters),
  celltype     = as.character(pbmc$celltype),
  row.names    = NULL), as.data.frame(emb, row.names = NULL)),
  file.path(FIG, "r_cell_meta.csv"), row.names = FALSE)

hvf <- HVFInfo(pbmc, method = "vst")
selected <- VariableFeatures(pbmc)
write.csv(data.frame(
  gene        = rownames(hvf),
  mean        = hvf[["mean"]],
  variance    = hvf[["variance"]],
  var.expected = hvf[["variance.expected"]],
  var.std     = hvf[["variance.standardized"]],
  # rank in the selection (1 = most variable), NA for the genes not selected
  hvg_rank    = match(rownames(hvf), selected)),
  file.path(FIG, "r_hvg.csv"), row.names = FALSE)

write.csv(data.frame(
  cluster    = as.character(all_markers$cluster),
  gene       = all_markers$gene,
  avg_log2FC = all_markers$avg_log2FC,
  pct.1      = all_markers$pct.1,
  pct.2      = all_markers$pct.2,
  p_val      = all_markers$p_val,
  p_val_adj  = all_markers$p_val_adj),
  file.path(FIG, "r_markers.csv"), row.names = FALSE)

cluster_sizes <- as.integer(table(pbmc$seurat_clusters))
names(cluster_sizes) <- levels(pbmc$seurat_clusters)
anchors <- list(
  n_genes_raw    = n_genes_raw,
  n_cells_raw    = n_cells_raw,
  n_cells_qc     = ncol(pbmc),
  n_hvg          = length(selected),
  n_clusters     = length(levels(pbmc$seurat_clusters)),
  n_markers      = nrow(all_markers),
  knn_nnz        = length(pbmc@graphs$RNA_nn@x),
  snn_nnz        = length(pbmc@graphs$RNA_snn@x),
  snn_weight_sum = sum(pbmc@graphs$RNA_snn@x),
  data_sum       = sum(GetAssayData(pbmc, layer = "data")),
  pc_stdev       = as.numeric(Stdev(pbmc, "pca"))[1:10],
  cluster_sizes  = as.list(cluster_sizes)
)
writeLines(jsonlite::toJSON(anchors, digits = 22, auto_unbox = TRUE, pretty = TRUE),
           file.path(FIG, "r_anchors.json"))
cat("Wrote r_cell_meta.csv, r_hvg.csv, r_markers.csv and r_anchors.json to", FIG, "\n")

# ---- RidgePlot (LYZ / NKG7 / MS4A1 / CD8A across clusters) -------------------
# The one figure in pbmc3k_tutorial.md without a canonical satijalab.org image;
# every other R panel there links the published vignette.
Idents(pbmc) <- pbmc$seurat_clusters
p <- RidgePlot(pbmc, features = c("LYZ", "NKG7", "MS4A1", "CD8A"), ncol = 2) &
  theme(plot.title = element_text(size = 11))
ggsave(file.path(FIG, "r_12_ridge_plot.png"), p, width = 11, height = 8, dpi = 150, bg = "white")
cat("wrote", file.path(FIG, "r_12_ridge_plot.png"), "\n")
