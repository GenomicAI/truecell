# Writes the segmentation-centroid reference: the point SeuratObject gives a cell
# polygon, for tests/test_anndata_spatial.py.
#
#   Rscript make_segmentation_centroid_reference.R <out.json>
#
# SeuratObject turns a Segmentation into Centroids with GetTissueCoordinates(full = FALSE),
# which is sp's label point for the cell's ring. Doubles are written as hex so they come
# back to the bit.
suppressPackageStartupMessages({ library(SeuratObject); library(sp); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript make_segmentation_centroid_reference.R <out.json>")
set.seed(11)

hex <- function(v) sprintf("%a", v)
polygons <- list()
add <- function(cell, x, y) polygons[[length(polygons) + 1]] <<- list(cell = cell, x = x, y = y)

# Star-shaped rings of 5-40 vertices spread over a slide. Half run anticlockwise, which sp
# stores reversed, and half clockwise.
for (k in 1:40) {
  n <- sample(5:40, 1)
  ang <- sort(runif(n, 0, 2 * pi))
  if (k %% 2 == 1) ang <- rev(ang)
  r <- runif(n, 2, 12)
  centre <- runif(2, 0, 8000)
  add(sprintf("star%02d", k), centre[1] + r * cos(ang), centre[2] + r * sin(ang))
}
# Already closed, as truecell stores a ring.
add("closed", c(0, 4, 4, 0, 0), c(0, 0, 3, 3, 0))
# Extra vertices along one edge: the mean vertex is (1.2, 0.6), the centroid (1, 1).
add("collinear_edge", c(0, 1, 2, 3, 0), c(0, 0, 0, 0, 3))
# Rings with no area.
add("flat", c(0, 1, 2), c(0, 1, 2))
add("two_vertices", c(5, 6), c(5, 7))
add("one_point", c(9, 9, 9), c(1, 1, 1))
add("single", 3, 4)

cells <- vapply(polygons, `[[`, "", "cell")
coords <- do.call(rbind, lapply(polygons, function(p) data.frame(x = p$x, y = p$y, cell = p$cell)))
seg <- suppressWarnings(CreateSegmentation(coords))
cen <- GetTissueCoordinates(seg, full = FALSE)
cen <- cen[match(cells, cen$cell), ]
stopifnot(identical(cen$cell, cells))
via_as <- GetTissueCoordinates(as(seg, "Centroids"))
via_as <- via_as[match(cells, via_as$cell), ]
stopifnot(identical(via_as$x, cen$x), identical(via_as$y, cen$y))

write_json(list(
  r = R.version.string, arch = R.version$arch,
  seuratobject = as.character(packageVersion("SeuratObject")),
  sp = as.character(packageVersion("sp")),
  polygons = lapply(polygons, function(p) list(cell = p$cell, x = I(hex(p$x)), y = I(hex(p$y)))),
  centroid_x = I(hex(cen$x)), centroid_y = I(hex(cen$y))
), args[1], auto_unbox = TRUE, pretty = TRUE)
