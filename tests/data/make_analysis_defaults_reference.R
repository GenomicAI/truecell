# Writes tests/data/r_analysis_defaults.json: what Seurat 5.5.1's own code does
# for the analysis defaults tests/test_analysis_defaults.py checks.
#
#   Rscript tests/data/make_analysis_defaults_reference.R tests/data/r_analysis_defaults.json
#
# Each piece runs Seurat's code rather than a restatement of it:
#   * phase   - CellCycleScoring's assignment function, lifted out of its body
#   * dim_heatmap_cells - DimHeatmap's cell selection: SeuratObject:::Top on one
#               embedding column, the negative half reversed back, unlisted
#   * scale_data - ScaleData on a dense matrix, the call RunMixscape makes when
#               slot = "scale.data" and PrepLDA makes before its PCA
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(jsonlite)
})

out_path <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out_path)) stop("usage: Rscript make_analysis_defaults_reference.R <out.json>")

ref <- list(
  r = paste(R.version$major, R.version$minor, sep = "."),
  seurat = as.character(packageVersion("Seurat")),
  seurat_object = as.character(packageVersion("SeuratObject"))
)

# ---- CellCycleScoring: the phase rule ----------------------------------------
body_calls <- as.list(body(Seurat::CellCycleScoring))
assign_call <- Filter(function(x) {
  is.call(x) && identical(x[[1]], as.name("<-")) && identical(x[[2]], as.name("assignments"))
}, body_calls)[[1]]
assign_phase <- eval(assign_call[[3]]$FUN)
pairs <- list(c(-1, -2), c(0, -1), c(-1, 0), c(0, 0), c(1, 1), c(0.5, 0.2),
              c(-0.5, 0.2), c(-1e-9, -2e-9))
ref$phase <- lapply(pairs, function(p) {
  list(s = p[1], g2m = p[2], phase = unname(assign_phase(p)))
})

# ---- DimHeatmap: which cells, in which order ---------------------------------
dim_heatmap_cells <- function(scores, ncells, balanced) {
  emb <- matrix(scores, ncol = 1,
                dimnames = list(paste0("c", seq_along(scores) - 1), "PC_1"))
  cells <- suppressWarnings(SeuratObject:::Top(data = emb, num = ncells, balanced = balanced))
  if (balanced) cells$negative <- rev(cells$negative)
  as.integer(sub("c", "", unlist(unname(cells))))
}
six <- c(0.3, -1.2, 2.5, -0.1, 0.9, -2.0)
cases <- list(
  list(six, 6, TRUE), list(six, 4, TRUE), list(six, 3, TRUE), list(six, 10, TRUE),
  list(six, 3, FALSE),
  list(c(0.4, -0.3, 1.1, -2.2, 0.05, 0.7, -0.9), 7, TRUE),   # odd, round(3.5) = 4
  list(c(0.4, -0.3, 1.1, -2.2, 0.05), 5, TRUE),              # odd, round(2.5) = 2
  list(c(3, 1, -1, 2, -2, 0.5, -0.5, 4, -4), 9, TRUE),       # odd, round(4.5) = 4
  list(c(1, 1, 0, -1, -1, 0.5), 6, TRUE),                     # tied scores
  list(c(1, -1, 0, 1, -1, 0.5), 4, FALSE))                    # tied |scores|
ref$dim_heatmap_cells <- lapply(cases, function(k) {
  list(scores = k[[1]], cells = k[[2]], balanced = k[[3]],
       order = dim_heatmap_cells(k[[1]], k[[2]], k[[3]]))
})

# ---- ScaleData on a matrix ----------------------------------------------------
# 150 cells so a single outlier scores past scale.max = 10; one row constant.
set.seed(7)
m <- matrix(round(rnorm(4 * 150, mean = 1, sd = 0.5), 3), nrow = 4,
            dimnames = list(paste0("g", 1:4), paste0("c", 1:150)))
m[2, ] <- 2
m[3, 1] <- 40
m[4, 2] <- -40
scaled <- ScaleData(m, features = rownames(m), verbose = FALSE)
ref$scale_data <- list(input = unname(m), scaled = unname(scaled))

writeLines(toJSON(ref, auto_unbox = TRUE, digits = 22, pretty = FALSE), out_path)
cat("wrote", out_path, "\n")
