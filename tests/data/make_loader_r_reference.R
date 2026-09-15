# What LoadNanostring, LoadVizgen and LoadXenium build from the synthetic bundles
# in tests/data/loaders, recorded for tests/test_loaders_vs_seurat.py.
#
#   python tests/data/make_loader_bundles.py
#   Rscript tests/data/make_loader_r_reference.R
#
# Needs Seurat, jsonlite, arrow (cells.parquet) and hdf5r (Vizgen boundaries).
# Without arrow, ReadXenium falls back to cells.csv.gz, which the parquet
# bundles leave out on purpose, so a missing package fails here rather than
# recording a different reader.

suppressPackageStartupMessages({
  library(Seurat)
  library(jsonlite)
})
stopifnot(requireNamespace("arrow", quietly = TRUE),
          requireNamespace("hdf5r", quietly = TRUE))

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
bundles <- file.path(here, "loaders")

describe <- function(obj, fov) {
  assay <- DefaultAssay(obj)
  meta <- obj[[]]
  image <- obj[[fov]]
  centroids <- GetTissueCoordinates(image[["centroids"]])
  list(
    assay = assay,
    assays = I(Assays(obj)),
    assay_features = lapply(setNames(Assays(obj), Assays(obj)),
                            function(a) I(rownames(obj[[a]]))),
    cells = I(colnames(obj)),
    features = I(rownames(obj)),
    # Rows are features and columns cells, as in the object.
    counts = unname(as.matrix(LayerData(obj, assay = assay, layer = "counts"))),
    n_count = I(unname(meta[[paste0("nCount_", assay)]])),
    n_feature = I(unname(meta[[paste0("nFeature_", assay)]])),
    meta_columns = I(colnames(meta)),
    segmentation_method = if ("segmentation_method" %in% colnames(meta)) {
      I(unname(meta$segmentation_method))
    },
    images = I(Images(obj)),
    image_cells = I(Cells(image)),
    boundaries = I(Boundaries(image)),
    default_boundary = DefaultBoundary(image),
    centroid_cells = I(centroids$cell),
    centroid_x = I(centroids$x),
    centroid_y = I(centroids$y),
    centroid_radius = Radius(image[["centroids"]])
  )
}

xenium <- function(name) {
  LoadXenium(file.path(bundles, name), fov = "fov", molecule.coordinates = FALSE)
}

reference <- list(
  seurat = as.character(packageVersion("Seurat")),
  seurat_object = as.character(packageVersion("SeuratObject")),
  cosmx = describe(LoadNanostring(file.path(bundles, "cosmx"), fov = "fov"), "fov"),
  merscope = describe(LoadVizgen(file.path(bundles, "merscope"), fov = "fov"), "fov"),
  xenium_csv = describe(xenium("xenium_csv"), "fov"),
  xenium_parquet = describe(xenium("xenium_parquet"), "fov"),
  xenium_parquet_binary = describe(xenium("xenium_parquet_binary"), "fov")
)

out <- file.path(here, "loader_r_reference.json")
write_json(reference, out, auto_unbox = TRUE, digits = 22, pretty = TRUE, null = "null")
cat("Wrote", out, "\n")
