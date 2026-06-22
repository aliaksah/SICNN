#!/usr/bin/env Rscript

results_dir <- file.path("rj_experiments", "optimizer_sweep", "kept_best_results")
out_file <- file.path(results_dir, "best_lbbnn_sicnn_summary_table.csv")

read_csv_if_exists <- function(path) {
  if (!file.exists(path)) {
    warning("Missing file: ", path, call. = FALSE)
    return(NULL)
  }
  read.csv(path, stringsAsFactors = FALSE)
}

first_existing <- function(paths) {
  hit <- paths[file.exists(paths)]
  if (length(hit) == 0) {
    return(NULL)
  }
  hit[[1]]
}

scalar <- function(x, default = NA) {
  if (length(x) == 0 || is.null(x)) {
    return(default)
  }
  x[[1]]
}

add_linear_rows <- function(rows) {
  linear_rho_grid <- read_csv_if_exists(file.path(
    results_dir,
    "rerun_forwardfix_n64000_rhogrid_m80_e2000_i5_lr0002_step500_eps1_steps200_summary.csv"
  ))

  if (!is.null(linear_rho_grid)) {
    for (i in seq_len(nrow(linear_rho_grid))) {
      rows[[length(rows) + 1]] <- data.frame(
        scenario = "linear",
        result_set = "paper-scale rho grid",
        rho = linear_rho_grid$rho[[i]],
        n_train = 64000,
        n_test = 8000,
        epochs = 2000,
        iter_per_epoch = 5,
        lr = 0.002,
        scheduler = "step, step_size=500",
        penalty = "80 * log(n_train)",
        epsilon_schedule = "1 -> 1e-5 over 200 steps",
        init = NA_character_,
        acc_sparse = linear_rho_grid$acc_sparse[[i]],
        acc_full = linear_rho_grid$acc_full[[i]],
        used_weights = linear_rho_grid$used_weights[[i]],
        exact_support_rate = linear_rho_grid$exact_support_rate[[i]],
        selected_support = NA_character_,
        note = "Best current linear setting; rho=0.9 selects one redundant feature in this run",
        stringsAsFactors = FALSE
      )
    }
  }

  rows
}

add_nonlinear_rows <- function(rows) {
  top40 <- read_csv_if_exists(file.path(
    results_dir,
    "lbbnn_nonlinear_overnight_discovery_top40.csv"
  ))

  if (!is.null(top40) && nrow(top40) > 0) {
    best <- top40[order(-top40$acc_sparse, top40$used_weights), ][1, ]
    rows[[length(rows) + 1]] <- data.frame(
      scenario = "nonlinear",
      result_set = "best overnight discovery",
      rho = best$rho,
      n_train = best$n_train,
      n_test = best$n_test,
      epochs = best$epochs,
      iter_per_epoch = NA_integer_,
      lr = best$lr,
      scheduler = paste0(best$scheduler_mode, ", step_size=", best$sch_step_size),
      penalty = paste0(best$penalty_mult, " * log(n_train)"),
      epsilon_schedule = paste0(best$epsilon_1, " -> ", best$epsilon_T, " over ", best$steps_T, " steps"),
      init = paste0(
        best$init_mode,
        "; hidden=", best$hidden_init_scale,
        ", covariate=", best$covariate_init_scale,
        ", direct=", best$direct_init_scale
      ),
      acc_sparse = best$acc_sparse,
      acc_full = best$acc_full,
      used_weights = best$used_weights,
      exact_support_rate = as.numeric(best$exact_support),
      selected_support = best$feature_set,
      note = "Best exact-support nonlinear discovery fit at rho=0",
      stringsAsFactors = FALSE
    )
  }

  validation <- read_csv_if_exists(file.path(
    results_dir,
    "lbbnn_nonlinear_overnight_validation_summary.csv"
  ))

  if (!is.null(validation) && nrow(validation) > 0) {
    for (i in seq_len(nrow(validation))) {
      rows[[length(rows) + 1]] <- data.frame(
        scenario = "nonlinear",
        result_set = "overnight validation partial",
        rho = validation$rho[[i]],
        n_train = 1000,
        n_test = 1000,
        epochs = 750,
        iter_per_epoch = NA_integer_,
        lr = 0.005,
        scheduler = "late",
        penalty = paste0(validation$penalty_mult[[i]], " * log(n_train)"),
        epsilon_schedule = "0.05 -> 0.005 over 100 steps",
        init = "lbbnn_like; hidden=0.5, covariate=1, direct=1",
        acc_sparse = sprintf(
          "%.3f (%.3f, %.3f)",
          validation$acc_sparse_median[[i]],
          validation$acc_sparse_min[[i]],
          validation$acc_sparse_max[[i]]
        ),
        acc_full = NA_character_,
        used_weights = sprintf(
          "%.1f (%.0f, %.0f)",
          validation$used_weights_median[[i]],
          validation$used_weights_min[[i]],
          validation$used_weights_max[[i]]
        ),
        exact_support_rate = validation$exact_support_mean[[i]],
        selected_support = NA_character_,
        note = "Validation was completed only for penalties 0.125 and 0.14 before the overnight controller stopped",
        stringsAsFactors = FALSE
      )
    }
  }

  rows
}

rows <- list()
rows <- add_linear_rows(rows)
rows <- add_nonlinear_rows(rows)

summary_table <- if (length(rows) == 0) {
  data.frame()
} else {
  do.call(rbind, rows)
}

write.csv(summary_table, out_file, row.names = FALSE)

cat("Wrote ", out_file, "\n", sep = "")
print(summary_table)
