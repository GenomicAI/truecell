# Writes a DESeq2 reference: Seurat 5.5.1's FindMarkers(test.use = "DESeq2") and
# DESeq2 1.52.0's intermediates, on one synthetic negative-binomial count matrix.
#
#   Rscript make_deseq2_reference.R <out.json>
#
# The matrix has 8 samples per group, a mean-dispersion trend with scatter, 40
# planted DE genes, one planted Cook's outlier (gene-050, sample s03) and one
# all-zero gene (gene-400). Seurat is run on it as it ships, and DESeq2DETest's
# steps are then repeated by hand, so every intermediate the Python side needs
# can be recorded and the by-hand p-values checked against Seurat's own.
suppressPackageStartupMessages({
  library(Seurat); library(SeuratObject); library(DESeq2); library(locfit); library(jsonlite)
})
out_path <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out_path)) stop("usage: Rscript make_deseq2_reference.R <out.json>")

set.seed(20260912)
n_genes <- 400; n_per <- 8
groups <- rep(c("A", "B"), each = n_per)
base <- exp(runif(n_genes, log(2), log(2000)))
disp <- (0.05 + 2 / sqrt(base)) * exp(rnorm(n_genes, 0, 0.3))
lfc <- rep(0, n_genes); lfc[1:40] <- rep(c(2, -2, 1, -1), each = 10)
sf <- exp(rnorm(2 * n_per, 0, 0.2))
counts <- sapply(seq_len(2 * n_per), function(j) {
  rnbinom(n_genes, mu = base * sf[j] * 2^(lfc * (groups[j] == "A")), size = 1 / disp)
})
dimnames(counts) <- list(sprintf("gene-%03d", seq_len(n_genes)), sprintf("s%02d", seq_len(2 * n_per)))
counts["gene-050", "s03"] <- counts["gene-050", "s03"] * 60L + 500L
counts["gene-400", ] <- 0L
storage.mode(counts) <- "integer"

# ---- Seurat, as shipped ------------------------------------------------------
obj <- suppressWarnings(CreateSeuratObject(counts = as(counts, "dgCMatrix")))
obj <- NormalizeData(obj, verbose = FALSE)
Idents(obj) <- setNames(factor(groups, levels = c("A", "B")), colnames(counts))
seu <- FindMarkers(obj, ident.1 = "A", ident.2 = "B", test.use = "DESeq2", verbose = FALSE)

# Which fold change the DESeq2 route reports: counts-based or data-based?
cnt <- as.matrix(GetAssayData(obj, layer = "counts")); dat <- as.matrix(GetAssayData(obj, layer = "data"))
a <- groups == "A"; b <- groups == "B"; g <- rownames(seu)
# Seurat 5 puts the pseudocount on the group sum, not the group mean.
fc_counts <- log2((rowSums(cnt[g, a]) + 1) / sum(a)) - log2((rowSums(cnt[g, b]) + 1) / sum(b))
fc_data <- log2((rowSums(expm1(dat[g, a])) + 1) / sum(a)) - log2((rowSums(expm1(dat[g, b])) + 1) / sum(b))

# ---- DESeq2DETest, step by step, on the features Seurat tested --------------
feats <- rownames(obj)[rownames(obj) %in% rownames(seu)]
m <- cnt[feats, c(colnames(counts)[a], colnames(counts)[b])]
info <- data.frame(group = factor(rep(c("Group1", "Group2"), each = n_per)), row.names = colnames(m))
dds <- DESeqDataSetFromMatrix(countData = m, colData = info, design = ~group)
dds <- estimateSizeFactors(dds)
dds <- estimateDispersions(dds, fitType = "local", quiet = TRUE)
dds <- nbinomWaldTest(dds, quiet = TRUE)
res <- results(dds, contrast = c("group", "Group1", "Group2"), alpha = 0.05)
mc <- mcols(dds)

# The local trend: DESeq2's own fit (locfit's default tree interpolation), and the
# same local regression evaluated directly at every gene's mean.
minDisp <- 1e-8
nz <- !mc$allZero
use <- mc$dispGeneEst[nz] > 100 * minDisp
means <- mc$baseMean[nz][use]; disps <- mc$dispGeneEst[nz][use]
keep <- disps >= minDisp * 10
d <- data.frame(logDisps = log(disps), logMeans = log(means))[keep, ]
w <- means[keep]
at <- log(mc$baseMean[nz])
fit_default <- locfit(logDisps ~ logMeans, data = d, weights = w)
trend_default <- exp(predict(fit_default, data.frame(logMeans = at)))
# A fit evaluated at explicit points cannot be predicted at new data (locfit
# refuses to interpolate that structure), so read each point's value off the fit.
ev_pts <- sort(unique(at))
fit_direct <- locfit(logDisps ~ logMeans, data = d, weights = w, ev = ev_pts)
# predict(where = "fitp") returns the value at each evaluation point, in ev order.
direct <- predict(fit_direct, where = "fitp")
stopifnot(length(direct) == length(ev_pts))
trend_direct <- exp(direct[match(at, ev_pts)])

pv <- res$pvalue; names(pv) <- rownames(res)
chk <- seu$p_val; names(chk) <- rownames(seu)
same_p <- all.equal(unname(pv[rownames(seu)]), unname(chk), tolerance = 0)

ref <- list(
  r = paste(R.version$major, R.version$minor, sep = "."),
  seurat = as.character(packageVersion("Seurat")),
  deseq2 = as.character(packageVersion("DESeq2")),
  locfit = as.character(packageVersion("locfit")),
  genes = I(rownames(counts)), samples = I(colnames(counts)), groups = I(groups),
  counts = unname(counts),
  seurat_markers = list(gene = I(rownames(seu)), p_val = seu$p_val, avg_log2FC = seu$avg_log2FC,
                        pct_1 = seu$pct.1, pct_2 = seu$pct.2, p_val_adj = seu$p_val_adj),
  fold_change_check = list(max_abs_diff_counts_formula = max(abs(seu$avg_log2FC - fc_counts)),
                           max_abs_diff_data_formula = max(abs(seu$avg_log2FC - fc_data))),
  by_hand_matches_seurat = isTRUE(same_p),
  tested_features = I(feats),
  internals = list(
    size_factors = unname(sizeFactors(dds)),
    min_disp = minDisp, max_disp = max(10, ncol(dds)),
    var_log_disp_ests = attr(dispersionFunction(dds), "varLogDispEsts"),
    disp_prior_var = attr(dispersionFunction(dds), "dispPriorVar"),
    base_mean = mc$baseMean, all_zero = mc$allZero, disp_gene_est = mc$dispGeneEst,
    disp_fit = mc$dispFit, trend_default = { x <- rep(NA_real_, length(feats)); x[nz] <- trend_default; x },
    trend_direct = { x <- rep(NA_real_, length(feats)); x[nz] <- trend_direct; x },
    disp_map = mc$dispMAP, dispersion = mc$dispersion, disp_outlier = mc$dispOutlier,
    max_cooks = mc$maxCooks, log2_fold_change = res$log2FoldChange, pvalue = res$pvalue))
# ---- A low-count matrix ------------------------------------------------------
# nbinomWaldTest floors each fitted mean at minmu = 0.5 before weighting it, so the
# standard error of a gene with small means depends on that floor. The first matrix
# has few such genes; this one is mostly them, with 20 high-count genes so the size
# factors (medians over zero-free genes) exist.
set.seed(912)
n_low <- 300; m_low <- 30
low_groups <- rep(c("A", "B"), each = m_low)
base_low <- exp(runif(n_low, log(0.05), log(5)))
base_low[1:20] <- exp(runif(20, log(20), log(80)))
disp_low <- 0.3 + 1 / sqrt(base_low)
lfc_low <- rep(0, n_low); lfc_low[21:60] <- rep(c(1.5, -1.5), each = 20)
sf_low <- exp(rnorm(2 * m_low, 0, 0.15))
low <- sapply(seq_len(2 * m_low), function(j) {
  rnbinom(n_low, mu = base_low * sf_low[j] * 2^(lfc_low * (low_groups[j] == "A")), size = 1 / disp_low)
})
dimnames(low) <- list(sprintf("low-%03d", seq_len(n_low)), sprintf("c%02d", seq_len(2 * m_low)))
storage.mode(low) <- "integer"
low <- low[rowSums(low) > 0, , drop = FALSE]
low_info <- data.frame(group = factor(rep(c("Group1", "Group2"), each = m_low)), row.names = colnames(low))
dds_low <- DESeqDataSetFromMatrix(countData = low, colData = low_info, design = ~group)
dds_low <- estimateSizeFactors(dds_low)
dds_low <- estimateDispersions(dds_low, fitType = "local", quiet = TRUE)
dds_low <- nbinomWaldTest(dds_low, quiet = TRUE)
res_low <- results(dds_low, contrast = c("group", "Group1", "Group2"), alpha = 0.05)
mu_low <- assays(dds_low)[["mu"]]
mc_low <- mcols(dds_low)
# Low means are also where the local trend climbs, and where dispersions reach
# DESeq2's cap of max(10, number of samples): 60 here.
ref$low_count <- list(
  genes = I(rownames(low)), samples = I(colnames(low)), groups = I(low_groups), counts = unname(low),
  lfc_se = res_low$lfcSE, stat = res_low$stat, pvalue = res_low$pvalue,
  frac_means_below_floor = mean(mu_low < 0.5, na.rm = TRUE),
  max_disp = max(10, ncol(dds_low)), disp_gene_est = mc_low$dispGeneEst, dispersion = mc_low$dispersion)
cat(sprintf("low-count matrix: %d genes x %d samples; %.1f %% of fitted means below 0.5; %d called at Bonferroni 0.05\n",
            nrow(low), ncol(low), 100 * ref$low_count$frac_means_below_floor,
            sum(pmin(1, res_low$pvalue * nrow(low)) < 0.05, na.rm = TRUE)))
cat(sprintf("low-count dispersions above 10: %d (max %.3f, cap %g)\n",
            sum(mc_low$dispersion > 10, na.rm = TRUE), max(mc_low$dispersion, na.rm = TRUE), ref$low_count$max_disp))

# ---- A flat-likelihood matrix ------------------------------------------------
# Eight samples a side, most genes at means under 3 with a small true dispersion,
# and size factors as uneven as a pseudobulk's. For many of these genes the
# likelihood barely changes as the dispersion falls towards zero. DESeq2 reports
# them below 1e-6, which leaves them out of the trend fit.
set.seed(20260913)
m_flat <- 8
flat_groups <- rep(c("A", "B"), each = m_flat)
base_flat <- c(exp(runif(40, log(50), log(500))), exp(runif(300, log(0.3), log(3))), exp(runif(40, log(5), log(50))))
disp_flat <- c(rep(0.05, 40), rep(0.03, 300), rep(0.05, 40))
lfc_flat <- c(rep(0, 340), rep(c(1.5, -1.5), each = 20))
sf_flat <- exp(rnorm(2 * m_flat, 0, 0.4))
flat <- sapply(seq_len(2 * m_flat), function(j) {
  rnbinom(length(base_flat), mu = base_flat * sf_flat[j] * 2^(lfc_flat * (flat_groups[j] == "A")), size = 1 / disp_flat)
})
dimnames(flat) <- list(sprintf("flat-%03d", seq_len(nrow(flat))), sprintf("p%02d", seq_len(2 * m_flat)))
storage.mode(flat) <- "integer"
flat <- flat[rowSums(flat) > 0, , drop = FALSE]
flat_info <- data.frame(group = factor(rep(c("Group1", "Group2"), each = m_flat)), row.names = colnames(flat))
dds_flat <- DESeqDataSetFromMatrix(countData = flat, colData = flat_info, design = ~group)
dds_flat <- estimateSizeFactors(dds_flat)
dds_flat <- estimateDispersions(dds_flat, fitType = "local", quiet = TRUE)
dds_flat <- nbinomWaldTest(dds_flat, quiet = TRUE)
res_flat <- results(dds_flat, contrast = c("group", "Group1", "Group2"), alpha = 0.05)
mc_flat <- mcols(dds_flat)
ref$flat <- list(
  genes = I(rownames(flat)), samples = I(colnames(flat)), groups = I(flat_groups), counts = unname(flat),
  disp_gene_est = mc_flat$dispGeneEst, disp_fit = mc_flat$dispFit, dispersion = mc_flat$dispersion,
  pvalue = res_flat$pvalue)
cat(sprintf("flat matrix: %d genes x %d samples; gene-wise below 1e-6: %d; %d called at Bonferroni 0.05\n",
            nrow(flat), ncol(flat), sum(mc_flat$dispGeneEst < 1e-6), sum(res_flat$pvalue * nrow(flat) < 0.05, na.rm = TRUE)))

writeLines(toJSON(ref, auto_unbox = TRUE, digits = 22, na = "null", pretty = FALSE), out_path)

cat(sprintf("wrote %s\n", out_path))
cat(sprintf("Seurat tested %d of %d genes; %d DE at p_val_adj < 0.05; NA p-values: %s\n",
            nrow(seu), n_genes, sum(seu$p_val_adj < 0.05, na.rm = TRUE), paste(rownames(seu)[is.na(seu$p_val)], collapse = ",")))
cat(sprintf("by-hand DESeq2DETest p-values identical to Seurat's: %s\n", isTRUE(same_p)))
cat(sprintf("avg_log2FC max|diff|: counts formula %.3g, data formula %.3g\n",
            ref$fold_change_check$max_abs_diff_counts_formula, ref$fold_change_check$max_abs_diff_data_formula))
cat(sprintf("DESeq2's dispFit vs default locfit prediction max|log diff| %.3g; vs direct evaluation %.3g\n",
            max(abs(log(mc$dispFit[nz]) - log(trend_default))), max(abs(log(mc$dispFit[nz]) - log(trend_direct)))))
cat(sprintf("max_disp %g, varLogDispEsts %.4f, dispPriorVar %.4f, dispersion outliers %d\n",
            ref$internals$max_disp, ref$internals$var_log_disp_ests, ref$internals$disp_prior_var, sum(mc$dispOutlier, na.rm = TRUE)))
