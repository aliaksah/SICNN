#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(torch)
  library(tibble)
  library(dplyr)
})

# Nonlinear sparse-recovery simulation for the SICNN paper.
#
# Examples:
#   Rscript rj_experiments/nonlinear_sim_paper.R --preset=smoke
#   Rscript rj_experiments/nonlinear_sim_paper.R --preset=timing --workers=4
#   Rscript rj_experiments/nonlinear_sim_paper.R --preset=paper --workers=8 --batch-jobs=8
#   Rscript rj_experiments/nonlinear_sim_paper.R --preset=paper --sic-penalty-mult=0.1
#
# The paper preset uses 100 reps. It omits SIC-CV by default.

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

parse_chr_vec <- function(x, default) {
  if (is.null(x)) return(default)
  trimws(strsplit(x, ",", fixed = TRUE)[[1]])
}

load_sicnn <- function() {
  if (requireNamespace("devtools", quietly = TRUE) && file.exists("DESCRIPTION")) {
    suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
  } else {
    suppressPackageStartupMessages(library(SICNN))
  }
}

nonlinear_signal <- function(x_mat) {
  if (ncol(x_mat) < 6L) {
    stop("The nonlinear data generator needs at least 6 active variables")
  }
  1.25 * sin(x_mat[, 1]) +
    0.75 * (x_mat[, 2]^2 - 1) +
    1.00 * x_mat[, 3] * x_mat[, 4] +
    0.75 * cos(2 * x_mat[, 5]) +
    0.50 * x_mat[, 6]
}

generate_nonlinear_data <- function(n, p, s, snr, rho, seed) {
  if (s != 6L) {
    stop("This paper nonlinear simulation currently uses s = 6")
  }
  set.seed(seed)

  if (rho == 0) {
    x_mat <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  } else {
    sigma <- outer(seq_len(p), seq_len(p), function(i, j) rho ^ abs(i - j))
    x_mat <- matrix(stats::rnorm(n * p), nrow = n, ncol = p) %*% chol(sigma)
  }

  signal <- as.numeric(nonlinear_signal(x_mat))
  signal <- signal - mean(signal)
  var_signal <- stats::var(signal)
  if (!is.finite(var_signal) || var_signal <= 0) {
    stop("Non-positive signal variance; check nonlinear generator")
  }
  noise <- stats::rnorm(n, sd = sqrt(var_signal / snr))

  sim_df <- as.data.frame(x_mat)
  colnames(sim_df) <- paste0("x", seq_len(p))
  sim_df$signal <- signal
  sim_df$y <- signal + noise

  list(data = sim_df, true_active = seq_len(s))
}

split_data <- function(sim_df, train_prop, seed) {
  set.seed(seed)
  n_train <- floor(nrow(sim_df) * train_prop)
  train_idx <- sample(seq_len(nrow(sim_df)), n_train)
  list(
    train = sim_df[train_idx, , drop = FALSE],
    test = sim_df[-train_idx, , drop = FALSE]
  )
}

make_loader <- function(sim_df, p, batch_size, shuffle) {
  x_mat <- as.matrix(sim_df[, seq_len(p), drop = FALSE])
  y_vec <- as.numeric(sim_df$y)
  ds <- torch::tensor_dataset(
    torch::torch_tensor(x_mat, dtype = torch::torch_float()),
    torch::torch_tensor(y_vec, dtype = torch::torch_float())
  )
  torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)
}

support_metrics <- function(selected, true_active) {
  p <- length(selected)
  true_set <- rep(FALSE, p)
  true_set[true_active] <- TRUE

  tp <- sum(selected & true_set)
  fp <- sum(selected & !true_set)
  fn <- sum(!selected & true_set)
  tn <- sum(!selected & !true_set)

  list(
    selected_count = sum(selected),
    tp = tp,
    fp = fp,
    fn = fn,
    tn = tn,
    tpr = tp / sum(true_set),
    fpr = fp / sum(!true_set),
    fdr = if (sum(selected) > 0) fp / sum(selected) else 0,
    exact_support = identical(as.logical(selected), as.logical(true_set))
  )
}

prediction_metrics <- function(y_true, y_hat, signal_true, train_y) {
  mse <- mean((y_true - y_hat)^2)
  signal_mse <- mean((signal_true - y_hat)^2)
  rmse <- sqrt(mse)
  tss <- sum((y_true - mean(train_y))^2)
  rss <- sum((y_true - y_hat)^2)
  r2 <- if (tss > 0) 1 - rss / tss else NA_real_
  list(test_mse = mse, test_rmse = rmse, test_r2 = r2, test_signal_mse = signal_mse)
}

fit_sicnn <- function(train_loader, n_train, p, hidden_sizes, epochs, lr,
                      sch_step_size, epsilon_1, epsilon_T, steps_T,
                      sic_threshold, penalty) {
  model <- SICNN_Net(
    problem_type = "regression",
    sizes = c(p, hidden_sizes, 1L),
    input_skip = TRUE,
    device = "cpu"
  )

  suppressMessages(train_SICNN(
    epochs = epochs,
    restarts = 1L,
    SICNN = model,
    lr = lr,
    train_dl = train_loader,
    device = "cpu",
    scheduler = "step",
    sch_step_size = sch_step_size,
    n_train = n_train,
    epsilon_1 = epsilon_1,
    epsilon_T = epsilon_T,
    steps_T = steps_T,
    sic_threshold = sic_threshold,
    penalty = penalty
  ))

  model
}

predict_sicnn_sparse <- function(model, test_loader, epsilon_T, sic_threshold) {
  model$compute_paths_input_skip(epsilon = epsilon_T, threshold = sic_threshold)
  model$eval()
  pred <- numeric(0)
  y_true <- numeric(0)
  torch::with_no_grad({
    coro::loop(for (b in test_loader) {
      pred_b <- model(b[[1]], sparse = TRUE)$squeeze()
      pred <- c(pred, as.numeric(pred_b$cpu()))
      y_true <- c(y_true, as.numeric(b[[2]]$cpu()))
    })
  })
  list(pred = pred, y_true = y_true)
}

select_sicnn_features <- function(model, p, epsilon_T, sic_threshold) {
  model$compute_paths_input_skip(epsilon = epsilon_T, threshold = sic_threshold)
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

run_sic_bic <- function(split, test_loader, cfg) {
  n_train <- nrow(split$train)
  train_loader <- make_loader(
    split$train, cfg$p,
    batch_size = min(n_train, cfg$batch_size),
    shuffle = TRUE
  )
  started <- proc.time()[[3]]
  model <- fit_sicnn(
    train_loader = train_loader,
    n_train = n_train,
    p = cfg$p,
    hidden_sizes = cfg$hidden_sizes,
    epochs = cfg$epochs,
    lr = cfg$lr,
    sch_step_size = cfg$sch_step_size,
    epsilon_1 = cfg$epsilon_1,
    epsilon_T = cfg$epsilon_T,
    steps_T = cfg$steps_T,
    sic_threshold = cfg$sic_threshold,
    penalty = cfg$sic_penalty_mult * log(n_train)
  )
  elapsed <- proc.time()[[3]] - started

  pred <- predict_sicnn_sparse(model, test_loader, cfg$epsilon_T, cfg$sic_threshold)
  selected <- select_sicnn_features(model, cfg$p, cfg$epsilon_T, cfg$sic_threshold)
  sic_counts <- model$sic_weight_counts(
    epsilon = cfg$epsilon_T,
    threshold = cfg$sic_threshold,
    active_paths = TRUE
  )

  list(
    method = sprintf("sic_%s_logn", format(cfg$sic_penalty_mult, scientific = FALSE, trim = TRUE)),
    penalty = cfg$sic_penalty_mult * log(n_train),
    y_hat = pred$pred,
    y_true = pred$y_true,
    selected = selected,
    active_weights = as.numeric(sic_counts["active"]),
    elapsed_seconds = elapsed
  )
}

run_lasso <- function(split, cfg, lambda_rule) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("glmnet is required for lasso baselines")
  }
  started <- proc.time()[[3]]
  x_train <- as.matrix(split$train[, seq_len(cfg$p), drop = FALSE])
  x_test <- as.matrix(split$test[, seq_len(cfg$p), drop = FALSE])

  fit <- glmnet::cv.glmnet(
    x = x_train,
    y = split$train$y,
    alpha = 1,
    family = "gaussian",
    standardize = FALSE,
    nfolds = cfg$lasso_folds
  )
  lambda <- if (lambda_rule == "min") fit$lambda.min else fit$lambda.1se
  beta_hat <- as.numeric(stats::coef(fit, s = lambda))[-1]
  y_hat <- as.numeric(stats::predict(fit, newx = x_test, s = lambda))
  elapsed <- proc.time()[[3]] - started

  list(
    method = paste0("lasso_", lambda_rule),
    penalty = lambda,
    y_hat = y_hat,
    y_true = split$test$y,
    selected = abs(beta_hat) > cfg$coef_threshold,
    active_weights = sum(abs(beta_hat) > cfg$coef_threshold),
    elapsed_seconds = elapsed
  )
}

run_lm_full <- function(split, cfg) {
  started <- proc.time()[[3]]
  train_design <- data.frame(y = split$train$y, split$train[, seq_len(cfg$p), drop = FALSE])
  test_design <- data.frame(split$test[, seq_len(cfg$p), drop = FALSE])
  fit <- stats::lm(y ~ ., data = train_design)
  y_hat <- as.numeric(stats::predict(fit, newdata = test_design))
  elapsed <- proc.time()[[3]] - started

  list(
    method = "lm_full",
    penalty = NA_real_,
    y_hat = y_hat,
    y_true = split$test$y,
    selected = rep(TRUE, cfg$p),
    active_weights = cfg$p,
    elapsed_seconds = elapsed
  )
}

run_rf <- function(split, cfg) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("randomForest is required for the random_forest baseline")
  }
  started <- proc.time()[[3]]
  train_design <- data.frame(y = split$train$y, split$train[, seq_len(cfg$p), drop = FALSE])
  test_design <- data.frame(split$test[, seq_len(cfg$p), drop = FALSE])
  fit <- randomForest::randomForest(
    y ~ .,
    data = train_design,
    ntree = cfg$rf_ntree,
    importance = TRUE
  )
  y_hat <- as.numeric(stats::predict(fit, newdata = test_design))
  imp <- randomForest::importance(fit, type = 1)
  imp_vec <- as.numeric(imp[, 1])
  selected <- rep(FALSE, cfg$p)
  selected[order(imp_vec, decreasing = TRUE)[seq_len(cfg$s)]] <- TRUE
  elapsed <- proc.time()[[3]] - started

  list(
    method = "random_forest_top_s",
    penalty = NA_real_,
    y_hat = y_hat,
    y_true = split$test$y,
    selected = selected,
    active_weights = cfg$s,
    elapsed_seconds = elapsed
  )
}

run_oracle_signal <- function(split, cfg) {
  started <- proc.time()[[3]]
  selected <- rep(FALSE, cfg$p)
  selected[seq_len(cfg$s)] <- TRUE
  list(
    method = "oracle_signal",
    penalty = NA_real_,
    y_hat = split$test$signal,
    y_true = split$test$y,
    selected = selected,
    active_weights = cfg$s,
    elapsed_seconds = proc.time()[[3]] - started
  )
}

run_one_method <- function(method, split, test_loader, cfg) {
  if (method == "sic_bic") return(run_sic_bic(split, test_loader, cfg))
  if (method == "lasso_min") return(run_lasso(split, cfg, "min"))
  if (method == "lasso_1se") return(run_lasso(split, cfg, "1se"))
  if (method == "lm_full") return(run_lm_full(split, cfg))
  if (method == "random_forest_top_s") return(run_rf(split, cfg))
  if (method == "oracle_signal") return(run_oracle_signal(split, cfg))
  stop("Unhandled method: ", method)
}

result_row <- function(fit_result, train_y, signal_true, true_active, scenario, rep_id) {
  pred <- prediction_metrics(fit_result$y_true, fit_result$y_hat, signal_true, train_y)
  supp <- support_metrics(fit_result$selected, true_active)

  tibble(
    n = scenario$n,
    p = scenario$p,
    s = scenario$s,
    snr = scenario$snr,
    rho = scenario$rho,
    rep = rep_id,
    method = fit_result$method,
    penalty = fit_result$penalty,
    test_mse = pred$test_mse,
    test_rmse = pred$test_rmse,
    test_r2 = pred$test_r2,
    test_signal_mse = pred$test_signal_mse,
    selected_count = supp$selected_count,
    tp = supp$tp,
    fp = supp$fp,
    fn = supp$fn,
    tn = supp$tn,
    tpr = supp$tpr,
    fpr = supp$fpr,
    fdr = supp$fdr,
    exact_support = supp$exact_support,
    active_weights = fit_result$active_weights,
    elapsed_seconds = fit_result$elapsed_seconds
  )
}

make_config <- function(args) {
  preset <- arg_value(args, "preset", "smoke")

  if (preset == "paper") {
    cfg <- list(
      n_values = c(500L, 1000L),
      p_values = c(15L, 50L),
      s_values = 6L,
      snrs = c(3, 5, 10),
      rhos = c(0, 0.5),
      reps = 100L,
      methods = c("sic_bic", "random_forest_top_s", "lasso_min", "lasso_1se", "lm_full", "oracle_signal"),
      epochs = 3000L
    )
  } else if (preset == "timing") {
    cfg <- list(
      n_values = 1000L,
      p_values = 15L,
      s_values = 6L,
      snrs = 3,
      rhos = 0,
      reps = 1L,
      methods = c("sic_bic", "random_forest_top_s", "lasso_min", "lasso_1se", "lm_full", "oracle_signal"),
      epochs = 3000L
    )
  } else if (preset == "smoke") {
    cfg <- list(
      n_values = 300L,
      p_values = 15L,
      s_values = 6L,
      snrs = 3,
      rhos = 0,
      reps = 1L,
      methods = c("sic_bic", "random_forest_top_s", "lasso_min", "lasso_1se", "lm_full", "oracle_signal"),
      epochs = 25L
    )
  } else {
    stop("Unknown preset. Use smoke, timing, or paper.")
  }

  cfg$preset <- preset
  cfg$n_values <- parse_int_vec(arg_value(args, "n-values"), cfg$n_values)
  cfg$p_values <- parse_int_vec(arg_value(args, "p-values"), cfg$p_values)
  cfg$s_values <- parse_int_vec(arg_value(args, "s-values"), cfg$s_values)
  cfg$snrs <- parse_num_vec(arg_value(args, "snrs"), cfg$snrs)
  cfg$rhos <- parse_num_vec(arg_value(args, "rhos"), cfg$rhos)
  cfg$reps <- as.integer(arg_value(args, "reps", cfg$reps))
  cfg$methods <- parse_chr_vec(arg_value(args, "methods"), cfg$methods)
  cfg$epochs <- as.integer(arg_value(args, "epochs", cfg$epochs))

  cfg$seed <- as.integer(arg_value(args, "seed", 20260616L))
  cfg$train_prop <- as.numeric(arg_value(args, "train-prop", 0.8))
  cfg$hidden_sizes <- parse_int_vec(arg_value(args, "hidden-sizes"), c(10L, 10L))
  cfg$lr <- as.numeric(arg_value(args, "lr", 0.002))
  cfg$sch_step_size <- as.integer(arg_value(args, "sch-step-size", 750L))
  cfg$epsilon_1 <- as.numeric(arg_value(args, "epsilon-1", 10))
  cfg$epsilon_T <- as.numeric(arg_value(args, "epsilon-T", 1e-5))
  cfg$steps_T <- as.integer(arg_value(args, "steps-T", 100L))
  cfg$sic_threshold <- as.numeric(arg_value(args, "sic-threshold", 0.5))
  cfg$batch_size <- as.integer(arg_value(args, "batch-size", 200L))
  cfg$test_batch_size <- as.integer(arg_value(args, "test-batch-size", 100L))
  cfg$lasso_folds <- as.integer(arg_value(args, "lasso-folds", 10L))
  cfg$coef_threshold <- as.numeric(arg_value(args, "coef-threshold", 1e-8))
  cfg$rf_ntree <- as.integer(arg_value(args, "rf-ntree", 300L))
  cfg$sic_penalty_mult <- as.numeric(arg_value(args, "sic-penalty-mult", 0.1))
  cfg$workers <- as.integer(arg_value(args, "workers", 8L))
  cfg$torch_threads <- as.integer(arg_value(args, "torch-threads", 1L))
  cfg$batch_jobs <- as.integer(arg_value(args, "batch-jobs", NA_integer_))
  cfg$out <- arg_value(
    args,
    "out",
    file.path("rj_experiments", paste0("nonlinear_sim_paper_", preset, "_results.rds"))
  )

  unknown_methods <- setdiff(
    cfg$methods,
    c("sic_bic", "random_forest_top_s", "lasso_min", "lasso_1se", "lm_full", "oracle_signal")
  )
  if (length(unknown_methods) > 0) {
    stop("Unknown method(s): ", paste(unknown_methods, collapse = ", "))
  }
  if ("random_forest_top_s" %in% cfg$methods && !requireNamespace("randomForest", quietly = TRUE)) {
    stop("randomForest is required for random_forest_top_s. Remove it from --methods or install randomForest.")
  }

  cfg
}

make_jobs <- function(scenarios, reps, seed) {
  jobs <- vector("list", nrow(scenarios) * reps)
  job_id <- 1L
  for (scenario_id in seq_len(nrow(scenarios))) {
    for (rep_id in seq_len(reps)) {
      jobs[[job_id]] <- list(
        job_id = job_id,
        scenario_id = scenario_id,
        scenario = scenarios[scenario_id, , drop = FALSE],
        rep_id = rep_id,
        data_seed = seed + scenario_id * 100000L + rep_id * 100L
      )
      job_id <- job_id + 1L
    }
  }
  jobs
}

run_one_job <- function(job, cfg, total_jobs) {
  suppressPackageStartupMessages({
    library(torch)
    library(tibble)
    library(dplyr)
  })
  load_sicnn()
  torch::torch_set_num_threads(cfg$torch_threads)

  scenario <- job$scenario
  cfg_job <- cfg
  cfg_job$p <- scenario$p
  cfg_job$s <- scenario$s

  split_seed <- job$data_seed + 1L
  torch::torch_manual_seed(job$data_seed)

  sim <- generate_nonlinear_data(
    n = scenario$n,
    p = scenario$p,
    s = scenario$s,
    snr = scenario$snr,
    rho = scenario$rho,
    seed = job$data_seed
  )
  split <- split_data(sim$data, cfg$train_prop, split_seed)
  test_loader <- make_loader(
    split$test,
    scenario$p,
    batch_size = min(nrow(split$test), cfg$test_batch_size),
    shuffle = FALSE
  )

  cat(sprintf(
    "Job %d/%d: scenario %d, rep %d | n=%d, p=%d, s=%d, snr=%g, rho=%g\n",
    job$job_id, total_jobs, job$scenario_id, job$rep_id,
    scenario$n, scenario$p, scenario$s, scenario$snr, scenario$rho
  ))

  rows <- vector("list", length(cfg$methods))
  for (method_index in seq_along(cfg$methods)) {
    method <- cfg$methods[[method_index]]
    method_seed <- job$data_seed + method_index
    set.seed(method_seed)
    torch::torch_manual_seed(method_seed)
    cat(sprintf("  Job %d/%d method: %s\n", job$job_id, total_jobs, method))

    fit_result <- run_one_method(method, split, test_loader, cfg_job)
    rows[[method_index]] <- result_row(
      fit_result = fit_result,
      train_y = split$train$y,
      signal_true = split$test$signal,
      true_active = sim$true_active,
      scenario = scenario,
      rep_id = job$rep_id
    )
    cat(sprintf(
      "    Job %d/%d %s done in %.1fs | MSE %.4f | signal MSE %.4f | TPR %.3f | FPR %.3f\n",
      job$job_id, total_jobs, method,
      rows[[method_index]]$elapsed_seconds,
      rows[[method_index]]$test_mse,
      rows[[method_index]]$test_signal_mse,
      rows[[method_index]]$tpr,
      rows[[method_index]]$fpr
    ))
  }

  dplyr::bind_rows(rows)
}

summarise_results <- function(results) {
  results |>
    group_by(method, n, p, s, snr, rho) |>
    summarise(
      reps = dplyr::n(),
      mean_mse = mean(test_mse),
      se_mse = stats::sd(test_mse) / sqrt(dplyr::n()),
      mean_signal_mse = mean(test_signal_mse),
      mean_r2 = mean(test_r2),
      mean_selected = mean(selected_count),
      mean_tpr = mean(tpr),
      mean_fpr = mean(fpr),
      mean_fdr = mean(fdr),
      exact_support_rate = mean(exact_support),
      mean_active_weights = mean(active_weights),
      mean_elapsed_seconds = mean(elapsed_seconds),
      .groups = "drop"
    )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  cfg <- make_config(args)

  load_sicnn()
  torch::torch_set_num_threads(cfg$torch_threads)

  scenarios <- expand.grid(
    n = cfg$n_values,
    p = cfg$p_values,
    s = cfg$s_values,
    snr = cfg$snrs,
    rho = cfg$rhos,
    stringsAsFactors = FALSE
  )
  scenarios <- scenarios[scenarios$s <= scenarios$p, , drop = FALSE]
  jobs <- make_jobs(scenarios, cfg$reps, cfg$seed)

  cfg$workers <- max(1L, min(as.integer(cfg$workers), length(jobs)))
  if (is.na(cfg$batch_jobs)) cfg$batch_jobs <- cfg$workers
  cfg$batch_jobs <- max(1L, as.integer(cfg$batch_jobs))

  cat(sprintf("Preset: %s\n", cfg$preset))
  cat(sprintf("Scenarios: %d | reps per scenario: %d | methods: %s\n",
              nrow(scenarios), cfg$reps, paste(cfg$methods, collapse = ", ")))
  cat(sprintf("Nominal method evaluations: %d\n", nrow(scenarios) * cfg$reps * length(cfg$methods)))
  cat(sprintf("Epochs per SICNN fit: %d | RF trees: %d\n", cfg$epochs, cfg$rf_ntree))
  cat(sprintf("Parallel workers: %d | torch threads per worker: %d\n", cfg$workers, cfg$torch_threads))
  cat(sprintf("Progress/save batch size: %d scenario-rep jobs\n", cfg$batch_jobs))
  cat(sprintf("Output path: %s\n\n", cfg$out))

  dir.create(dirname(cfg$out), recursive = TRUE, showWarnings = FALSE)
  start_all <- proc.time()[[3]]

  if (cfg$workers == 1L) {
    result_chunks <- vector("list", length(jobs))
    for (job_index in seq_along(jobs)) {
      result_chunks[[job_index]] <- run_one_job(jobs[[job_index]], cfg = cfg, total_jobs = length(jobs))
      partial <- dplyr::bind_rows(result_chunks[seq_len(job_index)])
      attr(partial, "config") <- cfg
      attr(partial, "elapsed_seconds") <- proc.time()[[3]] - start_all
      saveRDS(partial, cfg$out)
      elapsed_now <- proc.time()[[3]] - start_all
      rate <- job_index / max(elapsed_now, 1e-9)
      remaining <- (length(jobs) - job_index) / max(rate, 1e-9)
      cat(sprintf(
        "Progress: %d/%d jobs complete (%.1f%%). Elapsed %.1f min, ETA %.1f min. Partial saved.\n",
        job_index, length(jobs), 100 * job_index / length(jobs),
        elapsed_now / 60, remaining / 60
      ))
      flush.console()
    }
  } else {
    cluster <- parallel::makeCluster(cfg$workers, outfile = "")
    on.exit(parallel::stopCluster(cluster), add = TRUE)
    parallel::clusterExport(cluster, varlist = setdiff(ls(envir = .GlobalEnv), "args"), envir = .GlobalEnv)
    parallel::clusterCall(cluster, setwd, getwd())

    job_groups <- split(jobs, ceiling(seq_along(jobs) / cfg$batch_jobs))
    result_chunks <- list()
    completed_jobs <- 0L
    for (group_index in seq_along(job_groups)) {
      group <- job_groups[[group_index]]
      group_results <- parallel::parLapplyLB(
        cluster,
        group,
        run_one_job,
        cfg = cfg,
        total_jobs = length(jobs)
      )
      result_chunks <- c(result_chunks, group_results)
      completed_jobs <- completed_jobs + length(group)

      partial <- dplyr::bind_rows(result_chunks)
      attr(partial, "config") <- cfg
      attr(partial, "elapsed_seconds") <- proc.time()[[3]] - start_all
      saveRDS(partial, cfg$out)

      elapsed_now <- proc.time()[[3]] - start_all
      rate <- completed_jobs / max(elapsed_now, 1e-9)
      remaining <- (length(jobs) - completed_jobs) / max(rate, 1e-9)
      cat(sprintf(
        "Progress: %d/%d jobs complete (%.1f%%). Elapsed %.1f min, ETA %.1f min. Partial saved.\n",
        completed_jobs, length(jobs), 100 * completed_jobs / length(jobs),
        elapsed_now / 60, remaining / 60
      ))
      flush.console()
    }
  }

  results <- dplyr::bind_rows(result_chunks)
  elapsed_all <- proc.time()[[3]] - start_all
  attr(results, "config") <- cfg
  attr(results, "elapsed_seconds") <- elapsed_all
  saveRDS(results, cfg$out)

  summary_table <- summarise_results(results)
  overall_table <- results |>
    group_by(method) |>
    summarise(
      reps = dplyr::n(),
      mean_mse = mean(test_mse),
      se_mse = stats::sd(test_mse) / sqrt(dplyr::n()),
      mean_signal_mse = mean(test_signal_mse),
      mean_r2 = mean(test_r2),
      mean_selected = mean(selected_count),
      mean_tpr = mean(tpr),
      mean_fpr = mean(fpr),
      mean_fdr = mean(fdr),
      exact_support_rate = mean(exact_support),
      mean_active_weights = mean(active_weights),
      mean_elapsed_seconds = mean(elapsed_seconds),
      .groups = "drop"
    )
  write.csv(summary_table, sub("[.]rds$", "_by_scenario_summary.csv", cfg$out), row.names = FALSE)
  write.csv(overall_table, sub("[.]rds$", "_overall_summary.csv", cfg$out), row.names = FALSE)

  cat("\nSimulation complete.\n")
  cat(sprintf("Elapsed wall time: %.1f seconds (%.2f minutes)\n", elapsed_all, elapsed_all / 60))
  cat(sprintf("Final results saved to: %s\n\n", cfg$out))
  print(as.data.frame(overall_table[order(overall_table$mean_mse), ]))
}

main()
