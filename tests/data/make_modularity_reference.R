# Writes the modularity optimiser reference: Seurat's RunModularityClusteringCpp and
# FindClusters, for tests/test_modularity_seurat_parity.py.
#
#   Rscript make_modularity_reference.R <out.json>
#
# Every partition here comes out the same however Seurat's C++ is compiled. That rules
# out graphs whose weights tie exactly, such as small fractions: there an arm64 build,
# which fuses the multiply-add in the rule that moves a node, and an x86_64 build
# disagree with each other. truecell follows the x86_64 arithmetic.
suppressPackageStartupMessages({ library(Seurat); library(Matrix); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript make_modularity_reference.R <out.json>")
set.seed(1)

# A shared-nearest-neighbour graph as FindNeighbors builds it. Its weights are exactly
# inter / (2k - inter), so the JSON stores the integer counts.
k <- 10L
emb <- rbind(matrix(rnorm(500, 0), 50), matrix(rnorm(500, 3), 50),
             matrix(rnorm(500, c(0, 3)), 50), matrix(rnorm(500, -3), 50))
rownames(emb) <- paste0("c", seq_len(nrow(emb)))
snn <- as(FindNeighbors(emb, k.param = k, nn.method = "rann", verbose = FALSE)$snn, "dgCMatrix")
inter <- round(snn@x * 2 * k / (1 + snn@x))
stopifnot(identical(snn@x, inter / (2 * k - inter)))

# Not symmetric, with uniform weights: Seurat reads only the part below the diagonal.
asym <- rsparsematrix(100, 100, density = 0.06, rand.x = function(n) runif(n))

grid_runs <- function(A) {
  grid <- expand.grid(algorithm = 1:3, modularity_fxn = 1:2, random_seed = c(0L, 42L),
                      starts = c(1L, 10L))
  lapply(seq_len(nrow(grid)), function(r) {
    g <- grid[r, ]
    resolution <- c(0.8, 0.1)[g$modularity_fxn]
    labels <- Seurat:::RunModularityClusteringCpp(A, g$modularity_fxn, resolution, g$algorithm,
                                                  g$starts, g$starts, g$random_seed, FALSE, "")
    list(algorithm = g$algorithm, modularity_fxn = g$modularity_fxn, resolution = resolution,
         random_seed = g$random_seed, n_start = g$starts, n_iter = g$starts,
         labels = as.integer(labels))
  })
}

# java.util.Random's nextInt discards a draw that would bias it. With 3,000 nodes, seed 38
# first discards one at draw 820 of the first permutation; seed 0 never does there.
n <- 3000L
i <- rep(0:(n - 1L), each = 4L)
d <- rep(1:4, times = n)
j <- (i + d) %% n
w <- sqrt(1 + ((7L * i + d) %% 97L))
circulant <- sparseMatrix(i = c(i, j) + 1L, j = c(j, i) + 1L, x = c(w, w), dims = c(n, n))
rejection <- lapply(c(38L, 0L), function(seed) {
  labels <- Seurat:::RunModularityClusteringCpp(circulant, 1L, 0.8, 1L, 1L, 1L, seed, FALSE, "")
  list(random_seed = seed, labels = as.integer(labels))
})

# FindClusters itself, on the SNN graph plus three cells with no edges. The optimiser
# leaves each alone, and GroupSingletons, finding each tied across every cluster,
# places it with set.seed(1); sample().
cells <- c(colnames(snn), "iso1", "iso2", "iso3")
s <- summary(snn)
padded <- sparseMatrix(i = c(s$i, nrow(snn) + 1:3), j = c(s$j, nrow(snn) + 1:3),
                       x = c(s$x, rep(1, 3)), dims = rep(length(cells), 2),
                       dimnames = list(cells, cells))
find_clusters <- lapply(list(list(1L, TRUE), list(3L, TRUE), list(1L, FALSE)), function(a) {
  res <- FindClusters(padded, resolution = c(0.8, 1.6), algorithm = a[[1]],
                      group.singletons = a[[2]], verbose = FALSE)
  list(algorithm = a[[1]], group_singletons = a[[2]], resolution = c(0.8, 1.6),
       labels = unname(lapply(res, as.character)))
})

# GroupSingletons on hand-made cases.
singleton_case <- function(cells, ids, edges) {
  A <- Matrix(0, length(cells), length(cells), sparse = TRUE, dimnames = list(cells, cells))
  diag(A) <- 1
  for (e in edges) { A[e[[1]], e[[2]]] <- e[[3]]; A[e[[2]], e[[1]]] <- e[[3]] }
  A <- as(A, "dgCMatrix")
  names(ids) <- cells
  t <- summary(A)
  list(n = length(cells), i = t$i - 1L, j = t$j - 1L, x = t$x, ids = as.character(ids),
       grouped = as.character(Seurat:::GroupSingletons(ids, A, TRUE, verbose = FALSE)),
       ungrouped = as.character(Seurat:::GroupSingletons(ids, A, FALSE, verbose = FALSE)))
}
group_singletons <- list(
  # s2 is best connected to cluster 0 only once s1 has joined it: Seurat looks each
  # cluster's cells up again for every singleton.
  recomputed = singleton_case(
    c("a1", "a2", "s1", "b1", "b2", "b3", "s2"), c(0L, 0L, 2L, 1L, 1L, 1L, 3L),
    list(list("s1", "a1", 0.5), list("s1", "a2", 0.5), list("s2", "s1", 0.9),
         list("s2", "b1", 0.2), list("s2", "b2", 0.2), list("s2", "b3", 0.2),
         list("a1", "a2", 1), list("b1", "b2", 1), list("b2", "b3", 1))),
  # Cells with no edges tie across all five clusters, which first appear out of order.
  isolated = singleton_case(
    c("c3a", "c3b", "c0a", "iso1", "c0b", "c4a", "c4b", "c1a", "c1b", "c2a", "c2b", "iso2"),
    c(3L, 3L, 0L, 5L, 0L, 4L, 4L, 1L, 1L, 2L, 2L, 6L),
    list(list("c3a", "c3b", 1), list("c0a", "c0b", 1), list("c4a", "c4b", 1),
         list("c1a", "c1b", 1), list("c2a", "c2b", 1))),
  # Ten clusters tie, so R's draw (set.seed(1); sample.int(10, 1) is 9) takes the
  # ninth cluster to appear. For eight or fewer it takes the first.
  many = singleton_case(
    c(paste0(rep(letters[1:10], each = 2), 1:2), "iso"),
    c(rep(c(7L, 2L, 9L, 0L, 5L, 1L, 8L, 3L, 6L, 4L), each = 2), 10L),
    lapply(letters[1:10], function(l) list(paste0(l, 1), paste0(l, 2), 1))),
  # An exact tie between two non-zero means: 0.4 over two cells, 1.6 over four.
  tied = singleton_case(
    c("x1", "x2", "y1", "y2", "y3", "y4", "s"), c(1L, 1L, 0L, 0L, 0L, 0L, 2L),
    list(list("s", "x1", 0.4), list("s", "x2", 0.4), list("s", "y1", 0.8), list("s", "y2", 0.8),
         list("x1", "x2", 1), list("y1", "y2", 1), list("y3", "y4", 1), list("y2", "y3", 1)))
)

# The draw GroupSingletons breaks a tie with, for m tied clusters.
m <- c(1:300, 1000, 4095, 4096, 4097, 65535, 65536, 65537, 100000, 1000000)
draws <- sapply(m, function(size) { set.seed(1); sample.int(size, 1) })

out <- list(
  r = paste(R.version$major, R.version$minor, sep = "."),
  seurat = as.character(packageVersion("Seurat")),
  snn = list(n = nrow(snn), k = k, p = snn@p, i = snn@i, inter = as.integer(inter),
             runs = grid_runs(snn)),
  asym = list(n = nrow(asym), p = asym@p, i = asym@i, x = asym@x, runs = grid_runs(asym)),
  rejection = list(n = n, resolution = 0.8, runs = rejection),
  find_clusters = find_clusters,
  group_singletons = group_singletons,
  sample = list(m = m, draws = draws)
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 22), args[1])
cat(sprintf("wrote %s: Seurat %s, SNN %d cells / %d entries\n", args[1], out$seurat,
            nrow(snn), length(snn@x)))
