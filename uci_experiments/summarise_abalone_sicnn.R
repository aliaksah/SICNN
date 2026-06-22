files <- list.files("uci_experiments", full.names = TRUE)
files <- files[startsWith(basename(files), "abalone") & endsWith(files, "results.rds")]
rows <- lapply(files, function(f) {
  x <- as.data.frame(readRDS(f))
  x$source_file <- basename(f)
  x
})
cols <- unique(unlist(lapply(rows, names)))
rows <- lapply(rows, function(x) {
  for (m in setdiff(cols, names(x))) x[[m]] <- NA
  x[cols]
})
out <- do.call(rbind, rows)
out <- out[order(out$rmse_sparse_affine, out$used_weights, na.last = TRUE), ]
write.csv(out, "uci_experiments/abalone_sicnn_numeric_tuning_results.csv", row.names = FALSE)
keep <- intersect(c("source_file", "activation", "epochs", "lr", "scheduler", "sch_step_size", "penalty", "epsilon_1", "epsilon_T", "steps_T", "rmse_full", "rmse_sparse_affine", "corr_sparse", "pinball_sparse_affine", "used_weights", "avg_depth", "max_depth", "selected_features"), names(out))
print(head(out[keep], 25), row.names = FALSE)
