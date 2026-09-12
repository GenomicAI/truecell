# Writes tests/data/r_select_integration_features.json: Seurat 5.5.1's
# SelectIntegrationFeatures on hand-built variable-feature lists.
#
#   Rscript tests/data/make_select_integration_features_reference.R \
#     tests/data/r_select_integration_features.json
#
# Three objects whose lists tie on how many objects call a gene variable and on
# its median rank, with one gene (F29) missing from the first object, run at
# nfeatures values chosen so the cut lands inside those ties. R breaks a tie that
# survives both by the order `table()` puts names in, which is the collation of
# the session's locale; Rscript here runs with LC_COLLATE=C, recorded below.
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(jsonlite)
})

out_path <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out_path)) stop("usage: Rscript make_select_integration_features_reference.R <out.json>")

make_object <- function(features, variable) {
  counts <- matrix(1, nrow = length(features), ncol = 3,
                   dimnames = list(features, c("a", "b", "c")))
  obj <- suppressWarnings(CreateSeuratObject(counts = as(counts, "dgCMatrix")))
  VariableFeatures(obj) <- variable
  obj
}

all_features <- sprintf("F%02d", 1:30)
features <- list(setdiff(all_features, "F29"), all_features, all_features)
variable <- list(
  c("F05", "F02", "F09", "F01", "F12", "F07", "F03", "F20"),
  c("F02", "F05", "F11", "F09", "F14", "F01", "F21", "F03"),
  c("F09", "F02", "F13", "F05", "F01", "F22", "F07", "F29"))
objects <- Map(make_object, features, variable)

select <- function(objs, n) SelectIntegrationFeatures(object.list = objs, nfeatures = n, verbose = FALSE)

ref <- list(
  seurat = as.character(packageVersion("Seurat")),
  collate = Sys.getlocale("LC_COLLATE"),
  features = lapply(features, I),
  variable_features = lapply(variable, I),
  cases = lapply(c(1, 2, 4, 5, 6, 7, 8, 13, 20), function(n) list(nfeatures = n, selected = I(select(objects, n)))))

same <- list(make_object(all_features, c("F10", "F03", "F07")),
             make_object(all_features, c("F10", "F03", "F07")))
ref$identical_lists <- list(variable_features = I(c("F10", "F03", "F07")),
                            selected = I(select(same, 2000)))

mixed_names <- c("abc", "ABD", "Abc", "zeta")
mixed <- list(make_object(mixed_names, c("abc", "ABD")), make_object(mixed_names, c("ABD", "abc")))
ref$mixed_case <- list(features = I(mixed_names),
                       variable_features = list(I(c("abc", "ABD")), I(c("ABD", "abc"))),
                       nfeatures = 2, selected = I(select(mixed, 2)))

writeLines(toJSON(ref, auto_unbox = TRUE, pretty = TRUE), out_path)
cat("wrote", out_path, "\n")
for (k in ref$cases) cat(sprintf("  nfeatures=%-2d %s\n", k$nfeatures, paste(k$selected, collapse = " ")))
cat("  identical lists:", ref$identical_lists$selected, "| mixed case (", ref$collate, "):", ref$mixed_case$selected, "\n")
