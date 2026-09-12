# Writes the negative binomial reference: MASS::glm.nb, fitted as Seurat's
# GLMDETest fits it, on PBMC 3k genes where statsmodels' fit was unstable.
#
#   Rscript make_negbinom_reference.R <counts.json> <out.json>
#
# <counts.json> (negbinom_reference_counts.json) holds each gene's counts over the
# DE tutorial's clusters 0 and 1, with group 1 marking cluster 0's cells. It has 11
# of the 39 genes whose statsmodels p-value moved by more than 2 % between
# statsmodels 0.14.6 and 0.15.0, three well-fitted genes, and one whose groups
# separate (every count in one group is zero). Seurat's call is
# summary(glm.nb(GENE ~ group))$coef[2, 4], with group a factor of "Group1" and
# "Group2"; its coefficient is Group2 over Group1, so truecell's is its negative.
suppressPackageStartupMessages({ library(MASS); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript make_negbinom_reference.R <counts.json> <out.json>")

ref <- fromJSON(args[1])
group <- factor(ifelse(ref$group == 1, "Group1", "Group2"))
fits <- lapply(seq_along(ref$genes), function(i) {
  fit <- glm.nb(GENE ~ group, data = data.frame(GENE = ref$counts[i, ], group = group))
  co <- summary(fit)$coef
  list(gene = ref$genes[i], p_val = co[2, 4], coef_group2 = co[2, 1], se = co[2, 2], theta = fit$theta,
       theta_warning = if (is.null(fit$th.warn)) "" else fit$th.warn)
})
# The synthetic fixtures of tests/test_de_parity.py, stored with their counts.
synthetic <- lapply(seq_len(nrow(ref$synthetic)), function(k) {
  g <- factor(ifelse(ref$synthetic$group[[k]] == 1, "Group1", "Group2"))
  fit <- glm.nb(GENE ~ group, data = data.frame(GENE = ref$synthetic$counts[[k]], group = g))
  co <- summary(fit)$coef
  list(name = ref$synthetic$name[k], p_val = co[2, 4], coef_group2 = co[2, 1], se = co[2, 2], theta = fit$theta)
})
# With a covariate, as GLMDETest passes latent.vars: GENE ~ group + latent. The group
# means are then no longer the fitted means, so theta and the weights matter.
latent_fits <- lapply(ref$latent_genes, function(gene) {
  d <- data.frame(GENE = ref$counts[match(gene, ref$genes), ], group = group, latent = ref$latent)
  fit <- glm.nb(GENE ~ group + latent, data = d)
  co <- summary(fit)$coef
  list(gene = gene, p_val = co[2, 4], coef_group2 = co[2, 1], se = co[2, 2], theta = fit$theta,
       theta_warning = if (is.null(fit$th.warn)) "" else fit$th.warn)
})
out <- list(r = paste(R.version$major, R.version$minor, sep = "."),
            mass = as.character(packageVersion("MASS")), fits = fits, synthetic = synthetic,
            latent = latent_fits)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 22), args[2])
cat(sprintf("wrote %s: %d genes, MASS %s\n", args[2], length(fits), out$mass))
