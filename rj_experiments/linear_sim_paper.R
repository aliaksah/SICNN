#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(torch)
  library(tibble)
  library(dplyr)
})

# Linear sparse-recovery simulation for the SICNN paper.
#
# Examples:
#   Rscript rj_experiments/linear_sim_paper.R --preset=smoke
#   Rscript rj_experiments/linear_sim_paper.R --preset=timing
#   Rscript rj_experiments/linear_sim_paper.R --preset=paper
#   Rscript rj_experiments/linear_sim_paper.R --preset=paper --workers=8
#   Rscript rj_experiments/linear_sim_paper.R --preset=paper --methods=sic_bic,sic_cv,lasso_1se,lm_oracle --workers=4
#
# Useful overrides:
#   --methods=sic_bic,lasso_min,lasso_1se,lm_full,lm_oracle
#   --n-values=1000,2000 --p-values=15,50 --snrs=3,5,10 --rhos=0,0.5
#   --reps=20 --epochs=2000 --out=rj_experiments/linear_sim_paper_results.rds
#   --workers=8 --torch-threads=1

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) {
      next
    }
    stripped <- sub("^--", "", arg)
    if (grepl("=", stripped, fixed = TRUE)) {
      parts <- strsplit(stripped, "=", fixed = TRUE)[[1]]
      key <- parts[[1]]
      val <- paste(parts[-1], collapse = "=")
      out[[key]] <- val
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
  if (is.null(x)) {
    return(default)
  }
  as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])
}

parse_int_vec <- function(x, default) {
  as.integer(parse_num_vec(x, default))
}

parse_chr_vec <- function(x, default) {
  if (is.null(x)) {
    return(default)
  }
  trimws(strsplit(x, ",", fixed = TRUE)[[1]])
}

load_sicnn <- function() {
  if (requireNamespace("devtools", quietly = TRUE) && file.exists("DESCRIPTION")) {
    suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
  } else {
    suppressPackageStartupMessages(library(SICNN))
  }
}

make_beta <- function(p, s) {
  if (s > p) {
    stop("s must be less than or equal to p")
  }
  base <- c(0.60, -0.45, 0.40, -0.35, 0.30, -0.25, 0.20, -0.18, 0.15, -0.12)
  if (s <= length(base)) {
    beta_active <- base[seq_len(s)]
  } else {
    beta_active <- c(base, rep(0.10, s - length(base)))
  }
  c(beta_active, rep(0, p - s))
}

generate_linear_data <- function(n, p, s, snr, rho, seed) {
  set.seed(seed)
  beta_true <- make_beta(p, s)

  if (rho == 0) {
    x_mat <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  } else {
    sigma <- outer(seq_len(p), seq_len(p), function(i, j) rho ^ abs(i - j))
    chol_sigma <- chol(sigma)
    x_mat <- matrix(stats::rnorm(n * p), nrow = n, ncol = p) %*% chol_sigma
  }

  signal <- as.numeric(x_mat %*% beta_true)
  var_signal <- stats::var(signal)
  if (!is.finite(var_signal) || var_signal <= 0) {
    stop("Non-positive signal variance; check beta configuration")
  }
  noise <- stats::rnorm(n, sd = sqrt(var_signal / snr))

  sim_df <- as.data.frame(x_mat)
  colnames(sim_df) <- paste0("x", seq_len(p))
  sim_df$y <- signal + noise

  list(data = sim_df, beta_true = beta_true)
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

make_lm_design <- function(sim_df, p, cols) {
  if (length(cols) == 0) {
    return(data.frame(y = sim_df$y))
  }
  data.frame(y = sim_df$y, sim_df[, cols, drop = FALSE], check.names = FALSE)
}

support_metrics <- function(selected, true_active) {
  p <- length(selected)
  true_set <- rep(FALSE, p)
  true_set[true_active] <- TRUE

  tp <- sum(selected & true_set)
  fp <- sum(selected & !true_set)
  fn <- sum(!selected & true_set)
  tn <- sum(!selected & !true_set)

  tpr <- if (sum(true_set) > 0) tp / sum(true_set) else NA_real_
  fpr <- if (sum(!true_set) > 0) fp / sum(!true_set) else NA_real_
  fdr <- if (sum(selected) > 0) fp / sum(selected) else 0

  list(
    selected_count = sum(selected),
    tp = tp,
    fp = fp,
    fn = fn,
    tn = tn,
    tpr = tpr,
    fpr = fpr,
    fdr = fdr,
    exact_support = identical(as.logical(selected), as.logical(true_set))
  )
}

prediction_metrics <- function(y_true, y_hat, train_y) {
  mse <- mean((y_true - y_hat)^2)
  rmse <- sqrt(mse)
  tss <- sum((y_true - mean(train_y))^2)
  rss <- sum((y_true - y_hat)^2)
  r2 <- if (tss > 0) 1 - rss / tss else NA_real_
  list(test_mse = mse, test_rmse = rmse, test_r2 = r2)
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

estimate_sicnn_beta <- function(model, test_loader, beta_true, num_explain) {
  n_explain <- min(num_explain, length(test_loader$dataset$tensors[[2]]))
  coef_df <- coef(model, dataset = test_loader, num_data = n_explain)
  beta_hat <- as.numeric(coef_df$mean)
  sqrt(sum((beta_hat - beta_true)^2))
}

run_sic_bic <- function(split, test_loader, beta_true, cfg) {
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
    penalty = NULL
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
    method = "sic_bic",
    penalty = log(n_train),
    y_hat = pred$pred,
    y_true = pred$y_true,
    selected = selected,
    coef_error = estimate_sicnn_beta(model, test_loader, beta_true, cfg$num_explain),
    active_weights = as.numeric(sic_counts["active"]),
    elapsed_seconds = elapsed
  )
}

select_sic_lambda <- function(train_df, cfg, seed) {
  set.seed(seed)
  val_n <- max(1L, round(nrow(train_df) * cfg$cv_prop))
  val_idx <- sample(seq_len(nrow(train_df)), val_n)
  cv_train <- train_df[-val_idx, , drop = FALSE]
  cv_val <- train_df[val_idx, , drop = FALSE]

  cv_train_loader <- make_loader(
    cv_train, cfg$p,
    batch_size = min(nrow(cv_train), cfg$batch_size),
    shuffle = TRUE
  )
  cv_val_loader <- make_loader(
    cv_val, cfg$p,
    batch_size = min(nrow(cv_val), cfg$test_batch_size),
    shuffle = FALSE
  )

  cv_rmse <- numeric(length(cfg$lambda_grid))
  for (j in seq_along(cfg$lambda_grid)) {
    lambda <- cfg$lambda_grid[[j]]
    cat(sprintf("    CV lambda %d/%d: %.4f\n", j, length(cfg$lambda_grid), lambda))
    model <- fit_sicnn(
      train_loader = cv_train_loader,
      n_train = nrow(cv_train),
      p = cfg$p,
      hidden_sizes = cfg$hidden_sizes,
      epochs = cfg$epochs,
      lr = cfg$lr,
      sch_step_size = cfg$sch_step_size,
      epsilon_1 = cfg$epsilon_1,
      epsilon_T = cfg$epsilon_T,
      steps_T = cfg$steps_T,
      sic_threshold = cfg$sic_threshold,
      penalty = lambda
    )
    pred <- predict_sicnn_sparse(model, cv_val_loader, cfg$epsilon_T, cfg$sic_threshold)
    cv_rmse[[j]] <- sqrt(mean((pred$y_true - pred$pred)^2))
    cat(sprintf("      validation RMSE: %.4f\n", cv_rmse[[j]]))
  }

  best_idx <- which.min(cv_rmse)
  list(
    lambda = cfg$lambda_grid[[best_idx]],
    cv_rmse = cv_rmse[[best_idx]],
    all_rmse = cv_rmse
  )
}

run_sic_cv <- function(split, test_loader, beta_true, cfg, seed) {
  started <- proc.time()[[3]]
  cv <- select_sic_lambda(split$train, cfg, seed)
  cat(sprintf("    selected SIC lambda: %.4f\n", cv$lambda))

  n_train <- nrow(split$train)
  train_loader <- make_loader(
    split$train, cfg$p,
    batch_size = min(n_train, cfg$batch_size),
    shuffle = TRUE
  )
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
    penalty = cv$lambda
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
    method = "sic_cv",
    penalty = cv$lambda,
    y_hat = pred$pred,
    y_true = pred$y_true,
    selected = selected,
    coef_error = estimate_sicnn_beta(model, test_loader, beta_true, cfg$num_explain),
    active_weights = as.numeric(sic_counts["active"]),
    elapsed_seconds = elapsed
  )
}

run_lasso <- function(split, beta_true, cfg, lambda_rule) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("glmnet is required for lasso baselines")
  }
  started <- proc.time()[[3]]
  x_train <- as.matrix(split$train[, seq_len(cfg$p), drop = FALSE])
  y_train <- split$train$y
  x_test <- as.matrix(split$test[, seq_len(cfg$p), drop = FALSE])

  fit <- glmnet::cv.glmnet(
    x = x_train,
    y = y_train,
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
    coef_error = sqrt(sum((beta_hat - beta_true)^2)),
    active_weights = sum(abs(beta_hat) > cfg$coef_threshold),
    elapsed_seconds = elapsed
  )
}

run_lm_method <- function(split, beta_true, cfg, oracle) {
  started <- proc.time()[[3]]
  cols <- if (oracle) seq_len(cfg$s) else seq_len(cfg$p)
  train_design <- make_lm_design(split$train, cfg$p, cols)
  test_design <- make_lm_design(split$test, cfg$p, cols)
  fit <- stats::lm(y ~ ., data = train_design)
  y_hat <- as.numeric(stats::predict(fit, newdata = test_design))

  beta_hat <- rep(0, cfg$p)
  coefs <- stats::coef(fit)[-1]
  beta_hat[cols] <- as.numeric(coefs)

  selected <- rep(FALSE, cfg$p)
  selected[cols] <- TRUE
  elapsed <- proc.time()[[3]] - started

  list(
    method = if (oracle) "lm_oracle" else "lm_full",
    penalty = NA_real_,
    y_hat = y_hat,
    y_true = split$test$y,
    selected = selected,
    coef_error = sqrt(sum((beta_hat - beta_true)^2)),
    active_weights = sum(selected),
    elapsed_seconds = elapsed
  )
}

result_row <- function(fit_result, train_y, beta_true, scenario, rep_id) {
  pred <- prediction_metrics(fit_result$y_true, fit_result$y_hat, train_y)
  supp <- support_metrics(fit_result$selected, seq_len(scenario$s))

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
    coef_error = fit_result$coef_error,
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
  preset <- arg_value(args, "preset", "paper")

  if (preset == "paper") {
    cfg <- list(
      n_values = c(500L, 1000L),
      p_values = c(15L, 50L),
      s_values = 3L,
      snrs = c(3, 5, 10),
      rhos = c(0, 0.5),
      reps = 50L,
      methods = c("sic_bic", "lasso_min", "lasso_1se", "lm_full", "lm_oracle"),
      epochs = 2000L,
      lambda_grid = exp(seq(log(1), log(100), length.out = 10))
    )
  } else if (preset == "timing") {
    cfg <- list(
      n_values = 1000L,
      p_values = 15L,
      s_values = 3L,
      snrs = 3,
      rhos = 0,
      reps = 1L,
      methods = c("sic_bic", "lasso_min", "lasso_1se", "lm_full", "lm_oracle"),
      epochs = 2000L,
      lambda_grid = exp(seq(log(1), log(100), length.out = 10))
    )
  } else if (preset == "smoke") {
    cfg <- list(
      n_values = 300L,
      p_values = 15L,
      s_values = 3L,
      snrs = 3,
      rhos = 0,
      reps = 1L,
      methods = c("sic_bic", "lasso_min", "lasso_1se", "lm_full", "lm_oracle"),
      epochs = 25L,
      lambda_grid = exp(seq(log(1), log(20), length.out = 3))
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

  lambda_override <- arg_value(args, "lambda-grid")
  if (!is.null(lambda_override)) {
    cfg$lambda_grid <- parse_num_vec(lambda_override, cfg$lambda_grid)
  }

  cfg$seed <- as.integer(arg_value(args, "seed", 20260616L))
  cfg$train_prop <- as.numeric(arg_value(args, "train-prop", 0.8))
  cfg$cv_prop <- as.numeric(arg_value(args, "cv-prop", 0.1))
  cfg$hidden_sizes <- parse_int_vec(arg_value(args, "hidden-sizes"), c(5L, 5L))
  cfg$lr <- as.numeric(arg_value(args, "lr", 0.002))
  cfg$sch_step_size <- as.integer(arg_value(args, "sch-step-size", 500L))
  cfg$epsilon_1 <- as.numeric(arg_value(args, "epsilon-1", 10))
  cfg$epsilon_T <- as.numeric(arg_value(args, "epsilon-T", 1e-5))
  cfg$steps_T <- as.integer(arg_value(args, "steps-T", 100L))
  cfg$sic_threshold <- as.numeric(arg_value(args, "sic-threshold", 0.5))
  cfg$batch_size <- as.integer(arg_value(args, "batch-size", 200L))
  cfg$test_batch_size <- as.integer(arg_value(args, "test-batch-size", 100L))
  cfg$num_explain <- as.integer(arg_value(args, "num-explain", 50L))
  cfg$lasso_folds <- as.integer(arg_value(args, "lasso-folds", 10L))
  cfg$coef_threshold <- as.numeric(arg_value(args, "coef-threshold", 1e-8))
  cfg$workers <- as.integer(arg_value(args, "workers",12))
  cfg$torch_threads <- as.integer(arg_value(args, "torch-threads", 1L))
  cfg$out <- arg_value(
    args,
    "out",
    file.path("rj_experiments", paste0("linear_sim_paper_", preset, "_results.rds"))
  )

  unknown_methods <- setdiff(
    cfg$methods,
    c("sic_bic", "sic_cv", "lasso_min", "lasso_1se", "lm_full", "lm_oracle")
  )
  if (length(unknown_methods) > 0) {
    stop("Unknown method(s): ", paste(unknown_methods, collapse = ", "))
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

run_one_method <- function(method, split, test_loader, beta_true, cfg, seed) {
  if (method == "sic_bic") {
    return(run_sic_bic(split, test_loader, beta_true, cfg))
  }
  if (method == "sic_cv") {
    return(run_sic_cv(split, test_loader, beta_true, cfg, seed))
  }
  if (method == "lasso_min") {
    return(run_lasso(split, beta_true, cfg, "min"))
  }
  if (method == "lasso_1se") {
    return(run_lasso(split, beta_true, cfg, "1se"))
  }
  if (method == "lm_full") {
    return(run_lm_method(split, beta_true, cfg, oracle = FALSE))
  }
  if (method == "lm_oracle") {
    return(run_lm_method(split, beta_true, cfg, oracle = TRUE))
  }
  stop("Unhandled method: ", method)
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

  sim <- generate_linear_data(
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

    fit_result <- run_one_method(method, split, test_loader, sim$beta_true, cfg_job, method_seed)
    rows[[method_index]] <- result_row(
      fit_result = fit_result,
      train_y = split$train$y,
      beta_true = sim$beta_true,
      scenario = scenario,
      rep_id = job$rep_id
    )
    cat(sprintf(
      "    Job %d/%d %s done in %.1fs | MSE %.4f | TPR %.3f | FPR %.3f | selected %d\n",
      job$job_id, total_jobs, method,
      rows[[method_index]]$elapsed_seconds,
      rows[[method_index]]$test_mse,
      rows[[method_index]]$tpr,
      rows[[method_index]]$fpr,
      rows[[method_index]]$selected_count
    ))
  }

  dplyr::bind_rows(rows)
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

  if (is.na(cfg$workers)) {
    cfg$workers <- min(length(jobs), max(1L, parallel::detectCores(logical = FALSE) - 1L))
  }
  cfg$workers <- max(1L, min(as.integer(cfg$workers), length(jobs)))

  total_fits <- nrow(scenarios) * cfg$reps * length(cfg$methods)
  cat(sprintf("Preset: %s\n", cfg$preset))
  cat(sprintf("Scenarios: %d | reps per scenario: %d | methods: %s\n",
              nrow(scenarios), cfg$reps, paste(cfg$methods, collapse = ", ")))
  cat(sprintf("Nominal method evaluations: %d\n", total_fits))
  cat(sprintf("Epochs per SICNN fit: %d | lambda grid length: %d\n",
              cfg$epochs, length(cfg$lambda_grid)))
  cat(sprintf("Parallel workers: %d | torch threads per worker: %d\n",
              cfg$workers, cfg$torch_threads))
  cat(sprintf("Output path: %s\n\n", cfg$out))

  start_all <- proc.time()[[3]]
  if (cfg$workers == 1L) {
    result_chunks <- lapply(jobs, run_one_job, cfg = cfg, total_jobs = length(jobs))
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
  saveRDS(results, cfg$out)

  summary_table <- results |>
    group_by(method, n, p, s, snr, rho) |>
    summarise(
      reps = dplyr::n(),
      mean_mse = mean(test_mse),
      se_mse = stats::sd(test_mse) / sqrt(dplyr::n()),
      mean_r2 = mean(test_r2),
      mean_coef_error = mean(coef_error),
      mean_selected = mean(selected_count),
      mean_tpr = mean(tpr),
      mean_fpr = mean(fpr),
      mean_fdr = mean(fdr),
      exact_support_rate = mean(exact_support),
      mean_active_weights = mean(active_weights),
      mean_elapsed_seconds = mean(elapsed_seconds),
      .groups = "drop"
    )

  cat("\nSimulation complete.\n")
  cat(sprintf("Elapsed wall time: %.1f seconds (%.2f minutes)\n", elapsed_all, elapsed_all / 60))
  cat(sprintf("Final results saved to: %s\n\n", cfg$out))
  print(as.data.frame(summary_table))
}

main()
