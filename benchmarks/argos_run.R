# ARGOS (Egan, Li, Carvalho 2024, Commun. Phys.; CRAN package ARGOS 0.1.1, unmodified) on exported runs.
#
#   Rscript benchmarks/argos_run.R <manifest.tsv> <outdir> <sr_method: lasso|alasso> <num_samples>
#
# manifest.tsv: one line per run "key<TAB>path/to/X.csv<TAB>dt". For each run and each state variable:
# build_design_matrix(X, dt, sg_poly_order = 4, library_degree = 3, library_type = "poly") (package default
# smoothing: Savitzky-Golay order 4, window chosen by sg_optimal_combination), then argos(...) with the package
# defaults alpha_level = 0.05, num_samples (2000 = package default), ols_ps = TRUE, weights_method = NULL
# (ridge pilot for alasso), parallel = "no". Writes <outdir>/<key>.json with the identified model (term names
# -> coefficients, "(Intercept)" = constant term) and the in-R elapsed seconds (design matrix + 3 x argos).
suppressPackageStartupMessages({library(ARGOS); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
man <- read.table(args[1], sep = "\t", stringsAsFactors = FALSE, col.names = c("key", "csv", "dt"))
outdir <- args[2]; method <- args[3]; nsamp <- as.integer(args[4])
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
for (r in seq_len(nrow(man))) {
  key <- man$key[r]
  out <- file.path(outdir, paste0(key, ".json"))
  claim <- file.path(outdir, paste0(key, ".claim"))
  if (file.exists(out) || file.exists(claim)) next
  file.create(claim)
  X <- as.matrix(read.csv(man$csv[r], header = FALSE))
  set.seed(r)
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch({
    dm <- build_design_matrix(X, dt = man$dt[r], sg_poly_order = 4, library_degree = 3, library_type = "poly")
    lapply(1:ncol(X), function(j) {
      a <- argos(dm, library_type = "poly", state_var_deriv = j, alpha_level = 0.05, num_samples = nsamp,
                 sr_method = method, weights_method = NULL, ols_ps = TRUE, parallel = "no")
      m <- a$identified_model
      as.list(setNames(as.numeric(m[, 1]), rownames(m)))
    })
  }, error = function(e) conditionMessage(e))
  el <- proc.time()[["elapsed"]] - t0
  rec <- if (is.character(res)) list(key = key, error = res, seconds = el) else
    list(key = key, model = res, seconds = el, method = method, num_samples = nsamp)
  write(toJSON(rec, auto_unbox = TRUE, digits = NA), out)
  file.remove(claim)
}
