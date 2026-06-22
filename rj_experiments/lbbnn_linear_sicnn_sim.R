#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(torch)
  library(tibble)
  library(dplyr)
})

# SICNN replication of the LBBNN / ISLaB paper Section 3.2.1
# "Simulated linear data" experiment.
#
# Smoke test:
#   Rscript rj_experiments/lbbnn_linear_sicnn_sim.R --preset=smoke --workers=2
#
# Timing-sized run:
#   Rscript rj_experiments/lbbnn_linear_sicnn_sim.R --preset=timing --workers=4
#
# Paper-scale run, not for smoke testing:
#   Rscript rj_experiments/lbbnn_linear_sicnn_sim.R --preset=paper --workers=4

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    stripped <- sub("^--", "", arg)
    if (grepl("=", stripped, fixed = TRUE)) {
      parts <- strsplit(stripped, "=", fixed = TRUE)[[1]]
      out[[parts[[1]]]] <- paste(parts[-1], collapse = "=")
    } else {
      out[[stripped]] <- TRUE
    }
  }
  out
}

arg_value <- function(args, name, default = NULL) {
  if (!is.null(args[[name]])) args[[name]] else default
}

parse_num_vec <- function(x, default) {
  if (is.null(x)) return(default)
  as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])
}

parse_int_vec <- function(x, default) {
  as.integer(parse_num_vec(x, default))
}

parse_logical <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y")
}

load_sicnn <- function() {
  if (requireNamespace("devtools", quietly = TRUE) && file.exists("DESCRIPTION")) {
    suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
  } else {
    suppressPackageStartupMessages(library(SICNN))
  }
}

make_lbbnn_linear_data <- function(n_train, n_test, rho, noise_sd, seed) {
  set.seed(seed)
  n_total <- n_train + n_test

  x_mat <- matrix(stats::runif(n_total * 4L, min = -10, max = 10), ncol = 4L)
  x_mat[, 3L] <- rho * x_mat[, 1L] + (1 - rho) * x_mat[, 3L]
  colnames(x_mat) <- paste0("x", seq_len(4L))

  eta <- 100 + x_mat[, 1L] + x_mat[, 2L] + stats::rnorm(n_total, sd = noise_sd)
  y <- as.numeric(eta >= stats::median(eta))

  sim_df <- as.data.frame(x_mat)
  sim_df$eta <- eta
  sim_df$y <- y

  list(
    train = sim_df[seq_len(n_train), , drop = FALSE],
    test = sim_df[n_train + seq_len(n_test), , drop = FALSE],
    true_active = c(TRUE, TRUE, FALSE, FALSE)
  )
}

make_loader <- function(sim_df, batch_size, shuffle) {
  x_mat <- as.matrix(sim_df[, paste0("x", seq_len(4L)), drop = FALSE])
  y_vec <- as.numeric(sim_df$y)
  ds <- torch::tensor_dataset(
    torch::torch_tensor(x_mat, dtype = torch::torch_float()),
    torch::torch_tensor(y_vec, dtype = torch::torch_float())
  )
  torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)
}

make_batch_size <- function(n_train, iter_per_epoch, explicit_batch_size) {
  if (!is.na(explicit_batch_size)) return(as.integer(explicit_batch_size))
  max(1L, ceiling(n_train / iter_per_epoch))
}

fit_sicnn <- function(train_loader, cfg, n_train, seed) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  activation <- switch(
    cfg$activation,
    sigmoid = torch::nn_sigmoid(),
    relu = torch::nn_relu(),
    leaky_relu = torch::nn_leaky_relu(0.00),
    stop("Unknown activation: ", cfg$activation)
  )

  model <- SICNN_Net(
    problem_type = "binary classification",
    sizes = c(4L, cfg$hidden_sizes, 1L),
    input_skip = TRUE,
    device = "cpu",
    custom_act = activation
  )

  train_call <- function() {
    train_SICNN(
      epochs = cfg$epochs,
      restarts = cfg$restarts,
      SICNN = model,
      lr = cfg$lr,
      train_dl = train_loader,
      device = "cpu",
      scheduler = "step",
      sch_step_size = cfg$sch_step_size,
      n_train = n_train,
      epsilon_1 = cfg$epsilon_1,
      epsilon_T = cfg$epsilon_T,
      steps_T = cfg$steps_T,
      sic_threshold = cfg$sic_threshold,
      sic_threshold_type = cfg$sic_threshold_type,
      penalty = cfg$penalty
    )
  }

  if (isTRUE(cfg$show_epochs)) {
    train_call()
  } else {
    suppressMessages(train_call())
  }

  model
}

predict_binary_probs <- function(model, test_loader, sparse) {
  model$eval()
  prob <- numeric(0)
  y_true <- numeric(0)
  torch::with_no_grad({
    coro::loop(for (b in test_loader) {
      pred <- model(b[[1]], sparse = sparse)$squeeze()
      prob <- c(prob, as.numeric(pred$cpu()))
      y_true <- c(y_true, as.numeric(b[[2]]$cpu()))
    })
  })
  list(prob = prob, y_true = y_true)
}

binary_metrics <- function(y_true, prob, n_bins = 10L) {
  prob <- pmin(pmax(prob, 1e-8), 1 - 1e-8)
  pred_class <- as.numeric(prob >= 0.5)
  acc <- mean(pred_class == y_true)
  nll <- -mean(y_true * log(prob) + (1 - y_true) * log(1 - prob))

  confidence <- ifelse(prob >= 0.5, prob, 1 - prob)
  correct <- as.numeric(pred_class == y_true)
  breaks <- seq(0, 1, length.out = n_bins + 1L)
  bin_id <- cut(confidence, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  ece <- 0
  for (j in seq_len(n_bins)) {
    idx <- which(bin_id == j)
    if (length(idx) == 0L) next
    ece <- ece + length(idx) / length(y_true) * abs(mean(correct[idx]) - mean(confidence[idx]))
  }

  list(acc = acc, nll = nll, ece = ece)
}

select_sicnn_features <- function(model, p, epsilon_T, sic_threshold, sic_threshold_type) {
  model$compute_paths_input_skip(
    epsilon = epsilon_T,
    threshold = sic_threshold,
    threshold_type = sic_threshold_type
  )
  selected <- rep(FALSE, p)

  for (layer in model$layers$children) {
    alpha <- as.matrix(layer$alpha_active_path$cpu())
    in_features <- ncol(alpha)
    cov_cols <- if (in_features == p) seq_len(p) else (in_features - p + 1L):in_features
    selected <- selected | colSums(alpha[, cov_cols, drop = FALSE]) > 0
  }

  alpha_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  in_features <- ncol(alpha_out)
  cov_cols <- if (in_features == p) seq_len(p) else (in_features - p + 1L):in_features
  selected <- selected | colSums(alpha_out[, cov_cols, drop = FALSE]) > 0

  selected
}

path_depth_metrics <- function(model, p, epsilon_T, sic_threshold, sic_threshold_type) {
  model$compute_paths_input_skip(
    epsilon = epsilon_T,
    threshold = sic_threshold,
    threshold_type = sic_threshold_type
  )

  node_depths <- NULL
  for (layer_index in seq_along(model$layers$children)) {
    layer <- model$layers$children[[layer_index]]
    alpha <- as.matrix(layer$alpha_active_path$cpu())
    layer_depths <- vector("list", nrow(alpha))

    for (out_node in seq_len(nrow(alpha))) {
      depths <- integer(0)
      active_cols <- which(alpha[out_node, ] > 0)
      if (length(active_cols) > 0L) {
        for (col in active_cols) {
          if (is.null(node_depths) || col > length(node_depths)) {
            depths <- c(depths, 1L)
          } else if (length(node_depths[[col]]) > 0L) {
            depths <- c(depths, node_depths[[col]] + 1L)
          }
        }
      }
      layer_depths[[out_node]] <- depths
    }
    node_depths <- layer_depths
  }

  alpha_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  output_depths <- integer(0)
  for (out_node in seq_len(nrow(alpha_out))) {
    active_cols <- which(alpha_out[out_node, ] > 0)
    if (length(active_cols) == 0L) next
    for (col in active_cols) {
      if (is.null(node_depths) || col > length(node_depths)) {
        output_depths <- c(output_depths, 1L)
      } else if (length(node_depths[[col]]) > 0L) {
        output_depths <- c(output_depths, node_depths[[col]] + 1L)
      }
    }
  }

  if (length(output_depths) == 0L) {
    return(list(avg_depth = NA_real_, max_depth = NA_real_, n_active_paths = 0L))
  }

  list(
    avg_depth = mean(output_depths),
    max_depth = max(output_depths),
    n_active_paths = length(output_depths)
  )
}

support_metrics <- function(selected, true_active) {
  tp <- sum(selected & true_active)
  fp <- sum(selected & !true_active)
  fn <- sum(!selected & true_active)
  tn <- sum(!selected & !true_active)

  list(
    selected_count = sum(selected),
    tp = tp,
    fp = fp,
    fn = fn,
    tn = tn,
    tpr = tp / sum(true_active),
    fpr = fp / sum(!true_active),
    fdr = if (sum(selected) > 0) fp / sum(selected) else 0,
    exact_support = identical(as.logical(selected), as.logical(true_active))
  )
}

run_one_job <- function(job, cfg, total_jobs) {
  suppressPackageStartupMessages({
    library(torch)
    library(tibble)
    library(dplyr)
  })
  load_sicnn()
  torch::torch_set_num_threads(cfg$torch_threads)

  batch_size <- make_batch_size(cfg$n_train, cfg$iter_per_epoch, cfg$batch_size)
  test_batch_size <- min(cfg$n_test, cfg$test_batch_size)
  data_seed <- cfg$seed + job$rho_id * 100000L + job$rep * 100L
  fit_seed <- data_seed + 1L

  cat(sprintf(
    "Job %d/%d: rho=%g, rep=%d | n_train=%d, n_test=%d, epochs=%d\n",
    job$job_id, total_jobs, job$rho, job$rep, cfg$n_train, cfg$n_test, cfg$epochs
  ))

  sim <- make_lbbnn_linear_data(
    n_train = cfg$n_train,
    n_test = cfg$n_test,
    rho = job$rho,
    noise_sd = cfg$noise_sd,
    seed = data_seed
  )

  train_loader <- make_loader(sim$train, batch_size = batch_size, shuffle = TRUE)
  test_loader <- make_loader(sim$test, batch_size = test_batch_size, shuffle = FALSE)

  started <- proc.time()[[3]]
  model <- fit_sicnn(train_loader, cfg = cfg, n_train = cfg$n_train, seed = fit_seed)

  model$compute_paths_input_skip(
    epsilon = cfg$epsilon_T,
    threshold = cfg$sic_threshold,
    threshold_type = cfg$sic_threshold_type
  )

  dense_pred <- predict_binary_probs(model, test_loader, sparse = FALSE)
  sparse_pred <- predict_binary_probs(model, test_loader, sparse = TRUE)
  dense <- binary_metrics(dense_pred$y_true, dense_pred$prob, cfg$ece_bins)
  sparse <- binary_metrics(sparse_pred$y_true, sparse_pred$prob, cfg$ece_bins)

  selected <- select_sicnn_features(
    model,
    p = 4L,
    epsilon_T = cfg$epsilon_T,
    sic_threshold = cfg$sic_threshold,
    sic_threshold_type = cfg$sic_threshold_type
  )
  supp <- support_metrics(selected, sim$true_active)

  sic_counts <- model$sic_weight_counts(
    epsilon = cfg$epsilon_T,
    threshold = cfg$sic_threshold,
    threshold_type = cfg$sic_threshold_type,
    active_paths = TRUE
  )
  depth <- path_depth_metrics(
    model,
    p = 4L,
    epsilon_T = cfg$epsilon_T,
    sic_threshold = cfg$sic_threshold,
    sic_threshold_type = cfg$sic_threshold_type
  )

  elapsed <- proc.time()[[3]] - started
  cat(sprintf(
    "  Job %d/%d done in %.1fs | ACC full %.3f | ACC sparse %.3f | used weights %d | selected %s\n",
    job$job_id, total_jobs, elapsed, dense$acc, sparse$acc,
    as.integer(sic_counts[["active"]]),
    paste0(as.integer(selected), collapse = "")
  ))
  flush.console()

  tibble(
    method = "sicnn_smooth_l0",
    rho = job$rho,
    rep = job$rep,
    n_train = cfg$n_train,
    n_test = cfg$n_test,
    epochs = cfg$epochs,
    hidden_sizes = paste(cfg$hidden_sizes, collapse = "-"),
    penalty = if (is.null(cfg$penalty)) log(cfg$n_train) else cfg$penalty,
    epsilon_1 = cfg$epsilon_1,
    epsilon_T = cfg$epsilon_T,
    sic_threshold = cfg$sic_threshold,
    acc_full = dense$acc,
    acc_sparse = sparse$acc,
    ece_full = dense$ece,
    ece_sparse = sparse$ece,
    nll_full = dense$nll,
    nll_sparse = sparse$nll,
    used_weights = as.numeric(sic_counts[["active"]]),
    total_weights = as.numeric(sic_counts[["total"]]),
    avg_depth = depth$avg_depth,
    max_depth = depth$max_depth,
    n_active_paths = depth$n_active_paths,
    selected_count = supp$selected_count,
    tp = supp$tp,
    fp = supp$fp,
    fn = supp$fn,
    tn = supp$tn,
    tpr = supp$tpr,
    fpr = supp$fpr,
    fdr = supp$fdr,
    exact_support = supp$exact_support,
    inclusion_x1 = selected[[1L]],
    inclusion_x2 = selected[[2L]],
    inclusion_x3 = selected[[3L]],
    inclusion_x4 = selected[[4L]],
    elapsed_seconds = elapsed
  )
}

summarise_min_median_max <- function(results) {
  results |>
    group_by(method, rho) |>
    summarise(
      reps = dplyr::n(),
      acc_full = sprintf("%.3f (%.3f, %.3f)", median(acc_full), min(acc_full), max(acc_full)),
      acc_sparse = sprintf("%.3f (%.3f, %.3f)", median(acc_sparse), min(acc_sparse), max(acc_sparse)),
      used_weights = sprintf("%.1f (%.0f, %.0f)", median(used_weights), min(used_weights), max(used_weights)),
      avg_depth = sprintf("%.2f (%.2f, %.2f)", median(avg_depth, na.rm = TRUE), min(avg_depth, na.rm = TRUE), max(avg_depth, na.rm = TRUE)),
      max_depth = sprintf("%.1f (%.0f, %.0f)", median(max_depth, na.rm = TRUE), min(max_depth, na.rm = TRUE), max(max_depth, na.rm = TRUE)),
      ece_full = sprintf("%.3f (%.3f, %.3f)", median(ece_full), min(ece_full), max(ece_full)),
      ece_sparse = sprintf("%.3f (%.3f, %.3f)", median(ece_sparse), min(ece_sparse), max(ece_sparse)),
      nll_full = sprintf("%.3f (%.3f, %.3f)", median(nll_full), min(nll_full), max(nll_full)),
      nll_sparse = sprintf("%.3f (%.3f, %.3f)", median(nll_sparse), min(nll_sparse), max(nll_sparse)),
      mean_tpr = mean(tpr),
      mean_fpr = mean(fpr),
      exact_support_rate = mean(exact_support),
      mean_elapsed_seconds = mean(elapsed_seconds),
      .groups = "drop"
    )
}

summarise_inclusion_rates <- function(results) {
  results |>
    group_by(method, rho) |>
    summarise(
      reps = dplyr::n(),
      inclusion_rate_x1 = mean(inclusion_x1),
      inclusion_rate_x2 = mean(inclusion_x2),
      inclusion_rate_x3 = mean(inclusion_x3),
      inclusion_rate_x4 = mean(inclusion_x4),
      .groups = "drop"
    )
}

make_timing_estimates <- function(results, cfg, elapsed_all) {
  seconds_per_rho_rep <- mean(results$elapsed_seconds)
  timing_reps <- cfg$timing_reps
  rho_count <- length(cfg$rhos)

  tibble(
    reps_per_rho = timing_reps,
    rho_values = rho_count,
    total_rho_rep_jobs = timing_reps * rho_count,
    observed_mean_seconds_per_rho_rep = seconds_per_rho_rep,
    estimated_serial_minutes = seconds_per_rho_rep * timing_reps * rho_count / 60,
    estimated_parallel_minutes = seconds_per_rho_rep * ceiling(timing_reps * rho_count / cfg$workers) / 60,
    observed_smoke_wall_minutes = elapsed_all / 60
  )
}

make_jobs <- function(rhos, reps) {
  jobs <- vector("list", length(rhos) * reps)
  job_id <- 1L
  for (rho_id in seq_along(rhos)) {
    for (rep in seq_len(reps)) {
      jobs[[job_id]] <- list(
        job_id = job_id,
        rho_id = rho_id,
        rho = rhos[[rho_id]],
        rep = rep
      )
      job_id <- job_id + 1L
    }
  }
  jobs
}

make_config <- function(args) {
  preset <- arg_value(args, "preset", "paper")

  if (preset == "paper") {
    cfg <- list(
      n_train = 64000L,
      n_test = 8000L,
      rhos = c(0.0),
      reps = 10L,
      epochs = 2000L,
      iter_per_epoch = 50L,
      hidden_sizes = c(20L, 20L, 20L, 20L),
      lr = 0.002,
      sch_step_size = 500L,
      workers = 8L
    )
  } else if (preset == "timing") {
    cfg <- list(
      n_train = 4096L,
      n_test = 1024L,
      rhos = c(0.0, 0.1, 0.5, 0.9),
      reps = 2L,
      epochs = 50L,
      iter_per_epoch = 16L,
      hidden_sizes = c(20L, 20L, 20L, 20L),
      lr = 0.01,
      sch_step_size = 25L,
      workers = 4L
    )
  } else if (preset == "smoke") {
    cfg <- list(
      n_train = 256L,
      n_test = 128L,
      rhos = c(0.0, 0.9),
      reps = 1L,
      epochs = 3L,
      iter_per_epoch = 4L,
      hidden_sizes = c(20L, 20L, 20L, 20L),
      lr = 0.01,
      sch_step_size = 2L,
      workers = 2L
    )
  } else {
    stop("Unknown preset. Use smoke, timing, or paper.")
  }

  cfg$preset <- preset
  cfg$n_train <- as.integer(arg_value(args, "n-train", cfg$n_train))
  cfg$n_test <- as.integer(arg_value(args, "n-test", cfg$n_test))
  cfg$rhos <- parse_num_vec(arg_value(args, "rhos"), cfg$rhos)
  cfg$reps <- as.integer(arg_value(args, "reps", cfg$reps))
  cfg$epochs <- as.integer(arg_value(args, "epochs", cfg$epochs))
  cfg$iter_per_epoch <- as.integer(arg_value(args, "iter-per-epoch", cfg$iter_per_epoch))
  cfg$hidden_sizes <- parse_int_vec(arg_value(args, "hidden-sizes"), cfg$hidden_sizes)
  cfg$lr <- as.numeric(arg_value(args, "lr", cfg$lr))
  cfg$sch_step_size <- as.integer(arg_value(args, "sch-step-size", cfg$sch_step_size))
  cfg$workers <- as.integer(arg_value(args, "workers", cfg$workers))

  cfg$seed <- as.integer(arg_value(args, "seed", 20260617L))
  cfg$noise_sd <- as.numeric(arg_value(args, "noise-sd", 0.01))
  cfg$batch_size <- as.integer(arg_value(args, "batch-size", NA_integer_))
  cfg$test_batch_size <- as.integer(arg_value(args, "test-batch-size", 1024L))
  cfg$torch_threads <- as.integer(arg_value(args, "torch-threads", 1L))
  cfg$epsilon_1 <- as.numeric(arg_value(args, "epsilon-1", 10))
  cfg$epsilon_T <- as.numeric(arg_value(args, "epsilon-T", 1e-5))
  cfg$steps_T <- as.integer(arg_value(args, "steps-T", min(100L, cfg$epochs)))
  cfg$sic_threshold <- as.numeric(arg_value(args, "sic-threshold", 0.5))
  cfg$sic_threshold_type <- arg_value(args, "sic-threshold-type", "phi")
  cfg$penalty <- arg_value(args, "penalty", NULL)
  if (!is.null(cfg$penalty)) cfg$penalty <- as.numeric(cfg$penalty)
  cfg$restarts <- as.integer(arg_value(args, "restarts", 1L))
  cfg$activation <- arg_value(args, "activation", "sigmoid")
  cfg$ece_bins <- as.integer(arg_value(args, "ece-bins", 10L))
  cfg$show_epochs <- parse_logical(arg_value(args, "show-epochs", NULL), FALSE)
  cfg$timing_reps <- parse_int_vec(arg_value(args, "timing-reps"), c(1L, 2L, 5L, 10L))
  cfg$out <- arg_value(
    args,
    "out",
    file.path("rj_experiments", paste0("lbbnn_linear_sicnn_", preset, "_results.rds"))
  )

  if (!cfg$sic_threshold_type %in% c("phi", "abs")) {
    stop("sic-threshold-type must be 'phi' or 'abs'")
  }
  if (!cfg$activation %in% c("sigmoid", "relu", "leaky_relu")) {
    stop("activation must be one of: sigmoid, relu, leaky_relu")
  }
  if (length(cfg$hidden_sizes) < 1L) {
    stop("At least one hidden layer is required")
  }
  if (cfg$n_train < 2L || cfg$n_test < 1L) {
    stop("n-train and n-test must be positive, with n-train >= 2")
  }

  cfg
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  cfg <- make_config(args)

  load_sicnn()
  torch::torch_set_num_threads(cfg$torch_threads)

  jobs <- make_jobs(cfg$rhos, cfg$reps)
  cfg$workers <- max(1L, min(as.integer(cfg$workers), length(jobs)))

  cat("LBBNN Section 3.2.1 linear simulation with SICNN\n")
  cat(sprintf("Preset: %s\n", cfg$preset))
  cat(sprintf("rho values: %s | reps per rho: %d | jobs: %d\n",
              paste(cfg$rhos, collapse = ", "), cfg$reps, length(jobs)))
  cat(sprintf("n_train=%d | n_test=%d | epochs=%d | architecture=%s\n",
              cfg$n_train, cfg$n_test, cfg$epochs,
              paste(c(4L, cfg$hidden_sizes, 1L), collapse = "-")))
  cat(sprintf("activation=%s | penalty=%s | epsilon %.2g -> %.2g | threshold=%s %.3f\n",
              cfg$activation,
              if (is.null(cfg$penalty)) "log(n_train)" else as.character(cfg$penalty),
              cfg$epsilon_1, cfg$epsilon_T, cfg$sic_threshold_type, cfg$sic_threshold))
  cat(sprintf("Parallel workers: %d | torch threads per worker: %d\n", cfg$workers, cfg$torch_threads))
  cat(sprintf("Output path: %s\n\n", cfg$out))

  start_all <- proc.time()[[3]]
  if (cfg$workers == 1L) {
    result_chunks <- vector("list", length(jobs))
    for (job_index in seq_along(jobs)) {
      result_chunks[[job_index]] <- run_one_job(jobs[[job_index]], cfg = cfg, total_jobs = length(jobs))
      partial <- dplyr::bind_rows(result_chunks[seq_len(job_index)])
      attr(partial, "config") <- cfg
      attr(partial, "elapsed_seconds") <- proc.time()[[3]] - start_all
      dir.create(dirname(cfg$out), recursive = TRUE, showWarnings = FALSE)
      saveRDS(partial, cfg$out)
      cat(sprintf("Progress: %d/%d jobs complete. Partial saved.\n", job_index, length(jobs)))
      flush.console()
    }
  } else {
    cluster <- parallel::makeCluster(cfg$workers, outfile = "")
    on.exit(parallel::stopCluster(cluster), add = TRUE)
    parallel::clusterExport(
      cluster,
      varlist = setdiff(ls(envir = .GlobalEnv), "args"),
      envir = .GlobalEnv
    )
    parallel::clusterCall(cluster, setwd, getwd())
    result_chunks <- parallel::parLapplyLB(
      cluster,
      jobs,
      run_one_job,
      cfg = cfg,
      total_jobs = length(jobs)
    )
  }

  results <- dplyr::bind_rows(result_chunks)
  elapsed_all <- proc.time()[[3]] - start_all
  attr(results, "config") <- cfg
  attr(results, "elapsed_seconds") <- elapsed_all

  summary_table <- summarise_min_median_max(results)
  inclusion_table <- summarise_inclusion_rates(results)
  timing_table <- make_timing_estimates(results, cfg, elapsed_all)

  dir.create(dirname(cfg$out), recursive = TRUE, showWarnings = FALSE)
  saveRDS(results, cfg$out)
  write.csv(summary_table, sub("[.]rds$", "_summary.csv", cfg$out), row.names = FALSE)
  write.csv(inclusion_table, sub("[.]rds$", "_inclusion_rates.csv", cfg$out), row.names = FALSE)
  write.csv(timing_table, sub("[.]rds$", "_timing_estimates.csv", cfg$out), row.names = FALSE)

  cat("\nSimulation complete.\n")
  cat(sprintf("Elapsed wall time: %.1f seconds (%.2f minutes)\n", elapsed_all, elapsed_all / 60))
  cat(sprintf("Final results saved to: %s\n\n", cfg$out))
  print(as.data.frame(summary_table))
  cat("\nInclusion rates:\n")
  print(as.data.frame(inclusion_table))
  cat("\nTiming estimates based on this run:\n")
  print(as.data.frame(timing_table))
}

main()
