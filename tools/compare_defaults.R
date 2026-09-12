#!/usr/bin/env Rscript
# Dump the formals of the Seurat functions truecell ports, as JSON.
#
#   python tools/compare_defaults.py --refresh      # the way to run this
#   Rscript tools/compare_defaults.R <functions.json> <out.json>
#
# <functions.json> is an array of R expressions naming functions, such as
# "Seurat:::FindMarkers.default"; tools/compare_defaults.py writes it from its
# own map, so the two files cannot list different functions. Every formal is
# written as its deparsed default, or null when it has none, beside the package
# versions it was read from: a reference is only as good as the Seurat it came
# from.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: Rscript tools/compare_defaults.R <functions.json> <out.json>")
}
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(jsonlite)
})

deparse_default <- function(x) {
  # A formal with no default is the empty symbol; JSON has no such thing.
  if (is.symbol(x) && !nzchar(as.character(x))) {
    return(NA)
  }
  paste(deparse(x, width.cutoff = 500L), collapse = " ")
}

specs <- fromJSON(args[[1]])
functions <- list()
for (spec in specs) {
  f <- tryCatch(eval(parse(text = spec)), error = function(e) NULL)
  functions[[spec]] <- if (is.function(f)) lapply(formals(f), deparse_default) else "NOT FOUND"
}

out <- list(
  seurat = as.character(packageVersion("Seurat")),
  seurat_object = as.character(packageVersion("SeuratObject")),
  r = paste(R.version$major, R.version$minor, sep = "."),
  functions = functions
)
write(toJSON(out, auto_unbox = TRUE, na = "null", null = "null", pretty = TRUE), args[[2]])
