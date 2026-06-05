library(torch)
library(dplyr)
library(tidyr)
library(purrr)

# ── Data generation ───────────────────────────────────────────────────────────

generate_linear_data <- function(n, p, snr, beta_true) {
  X <- matrix(rnorm(n * p), ncol = p)
  signal <- X %*% beta_true
  var_signal <- var(signal)
  var_noise <- var_signal / snr
  noise <- rnorm(n, sd = sqrt(var_noise))
  y <- signal + noise
  data <- as.data.frame(X)
  colnames(data) <- paste0("x", 1:p)
  data$y <- as.numeric(y)
  return(data)
}

# ── CV helper: pick best lambda via a single held-out validation fold ─────────
# Splits the training set 88.9% / 11.1% (≈ 8:1 within the 80% train split,
# giving roughly a 70/10/20 overall split).  Trains one model per lambda
# candidate and returns the lambda with the lowest sparse-model RMSE.

cv_lambda <- function(data, lambda_grid, p, n_train_full,
                      epochs, lr, sch_step_size, sizes,
                      epsilon_1, epsilon_T, steps_T, sic_threshold) {

  n_total <- nrow(data)
  # Hold out ~11% of the full data as a CV validation fold
  cv_val_idx  <- sample(n_total, size = round(n_total * 0.10))
  cv_train_df <- data[-cv_val_idx, ]
  cv_val_df   <- data[ cv_val_idx, ]

  n_cv_train <- nrow(cv_train_df)

  cv_train_loaders <- get_dataloaders(
    cv_train_df,
    train_proportion  = 1.0,
    train_batch_size  = min(n_cv_train, 200L),
    test_batch_size   = 1L,       # unused
    standardize       = FALSE
  )
  cv_val_loaders <- get_dataloaders(
    cv_val_df,
    train_proportion  = 1.0,
    train_batch_size  = 1L,       # unused
    test_batch_size   = min(nrow(cv_val_df), 100L),
    standardize       = FALSE
  )

  cv_rmse <- numeric(length(lambda_grid))

  for (j in seq_along(lambda_grid)) {
    lam <- lambda_grid[j]
    cat(sprintf("  [CV] lambda %d/%d = %.3f\n", j, length(lambda_grid), lam))

    cv_model <- SICNN_Net(
      problem_type = "regression",
      sizes        = sizes,
      input_skip   = TRUE,
      device       = "cpu"
    )

    train_SICNN(
      epochs       = epochs,
      restarts     = 1,
      SICNN        = cv_model,
      lr           = lr,
      train_dl     = cv_train_loaders$train_loader,
      device       = "cpu",
      scheduler    = "step",
      sch_step_size = sch_step_size,
      n_train      = n_cv_train,
      epsilon_1    = epsilon_1,
      epsilon_T    = epsilon_T,
      steps_T      = steps_T,
      sic_threshold = sic_threshold,
      penalty      = lam
    )

    val_res <- validate_SICNN(
      cv_model,
      test_dl    = cv_val_loaders$train_loader,   # validation fold
      device     = "cpu",
      verbose    = FALSE
    )
    cv_rmse[j] <- as.numeric(val_res$validation_error_sparse)
  }

  best_idx <- which.min(cv_rmse)
  cat(sprintf("  [CV] best lambda = %.3f (RMSE = %.4f)\n",
              lambda_grid[best_idx], cv_rmse[best_idx]))

  list(
    best_lambda = lambda_grid[best_idx],
    best_rmse   = cv_rmse[best_idx],
    all_lambdas = lambda_grid,
    all_rmse    = cv_rmse
  )
}

# ── Simulation settings ───────────────────────────────────────────────────────

ns_list   <- c(1000)
snrs_list <- c(3, 5, 10)
n_reps    <- 5
p         <- 15
beta_true <- c(0.6, -0.4, 0.5, rep(0, p - 3))

# 10 log-spaced lambda candidates from 1 to 100
# BIC default at n_train = 800 is log(800) ≈ 6.9 — well inside this range
lambda_grid <- exp(seq(log(1), log(100), length.out = 10))
cat("Lambda grid:\n")
print(round(lambda_grid, 3))

# Shared training hyper-parameters
EPOCHS        <- 2000
LR            <- 0.002
SCH_STEP_SIZE <- 500
SIZES         <- c(p, 5, 5, 1)
EPSILON_1     <- 1
EPSILON_T     <- 1e-5
STEPS_T       <- 200
SIC_THRESHOLD <- 0.5

set.seed(42)
torch::torch_manual_seed(42)

# ── Experiment grid ───────────────────────────────────────────────────────────

experiments <- expand.grid(
  n      = ns_list,
  snr    = snrs_list,
  rep    = 1:n_reps,
  method = c("bic", "cv"),
  stringsAsFactors = FALSE
)

cat(sprintf("\nStarting simulation with %d total runs...\n", nrow(experiments)))

results <- vector("list", nrow(experiments))

# ── Helper: extract metrics from a fitted model ───────────────────────────────

extract_metrics <- function(model, loaders, n, snr, rep_id, method, penalty_used, p, beta_true) {

  val_res  <- validate_SICNN(model, test_dl = loaders$test_loader,
                              device = "cpu", verbose = FALSE)
  test_mse <- as.numeric(val_res$validation_error_sparse)^2  # RMSE -> MSE

  cf       <- coef(model, dataset = loaders$test_loader,
                   num_data = 10, num_samples = 1)
  beta_hat <- cf$mean
  coef_error <- sqrt(sum((beta_hat - beta_true)^2))

  # Feature selection via active paths
  model$compute_paths_input_skip(epsilon = EPSILON_T, threshold = SIC_THRESHOLD)

  selected <- rep(FALSE, p)
  for (l in model$layers$children) {
    alp      <- as.matrix(l$alpha_active_path$cpu())
    in_f     <- ncol(alp)
    cov_cols <- if (in_f == p) 1:p else (in_f - p + 1):in_f
    selected <- selected | (colSums(alp[, cov_cols, drop = FALSE]) > 0)
  }
  alp_out  <- as.matrix(model$out_layer$alpha_active_path$cpu())
  in_f     <- ncol(alp_out)
  cov_cols <- if (in_f == p) 1:p else (in_f - p + 1):in_f
  selected <- selected | (colSums(alp_out[, cov_cols, drop = FALSE]) > 0)

  true_vars  <- 1:3
  false_vars <- 4:p
  tpr <- sum(selected[true_vars]) / length(true_vars)
  fpr <- sum(selected[false_vars]) / length(false_vars)

  sic_counts    <- model$sic_weight_counts(epsilon = EPSILON_T, threshold = SIC_THRESHOLD, active_paths = TRUE)
  active_weights <- as.numeric(sic_counts["active"])

  tibble(
    n              = n,
    snr            = snr,
    rep            = rep_id,
    method         = method,
    penalty_used   = penalty_used,
    test_mse       = test_mse,
    coef_error     = coef_error,
    tpr            = tpr,
    fpr            = fpr,
    active_weights = active_weights
  )
}

# ── Main loop ─────────────────────────────────────────────────────────────────

for (i in seq_len(nrow(experiments))) {
  exp_i  <- experiments[i, ]
  method <- exp_i$method

  cat(sprintf(
    "\nRun %d/%d: N=%d, SNR=%d, Rep=%d, Method=%s\n",
    i, nrow(experiments), exp_i$n, exp_i$snr, exp_i$rep, method
  ))

  # Generate data
  data <- generate_linear_data(exp_i$n, p, exp_i$snr, beta_true)

  # Shared data loaders (80/20 train-test)
  loaders <- get_dataloaders(
    data,
    train_proportion  = 0.8,
    train_batch_size  = min(as.integer(exp_i$n * 0.8), 200L),
    test_batch_size   = min(as.integer(exp_i$n * 0.2), 100L),
    standardize       = FALSE
  )
  n_train <- as.integer(exp_i$n * 0.8)

  # ── Method: BIC (penalty = NULL → log(n_train)) ──────────────────────────
  if (method == "bic") {
    penalty_used <- log(n_train)

    model <- SICNN_Net(
      problem_type = "regression",
      sizes        = SIZES,
      input_skip   = TRUE,
      device       = "cpu"
    )

    train_SICNN(
      epochs        = EPOCHS,
      restarts      = 1,
      SICNN         = model,
      lr            = LR,
      train_dl      = loaders$train_loader,
      device        = "cpu",
      scheduler     = "step",
      sch_step_size = SCH_STEP_SIZE,
      n_train       = n_train,
      epsilon_1     = EPSILON_1,
      epsilon_T     = EPSILON_T,
      steps_T       = STEPS_T,
      sic_threshold = SIC_THRESHOLD,
      penalty       = NULL       # BIC default
    )

  # ── Method: CV (penalty chosen from lambda_grid) ─────────────────────────
  } else if (method == "cv") {
    cat("  Running CV to select lambda...\n")
    cv_out <- cv_lambda(
      data          = data,
      lambda_grid   = lambda_grid,
      p             = p,
      n_train_full  = n_train,
      epochs        = EPOCHS,
      lr            = LR,
      sch_step_size = SCH_STEP_SIZE,
      sizes         = SIZES,
      epsilon_1     = EPSILON_1,
      epsilon_T     = EPSILON_T,
      steps_T       = STEPS_T,
      sic_threshold = SIC_THRESHOLD
    )
    penalty_used <- cv_out$best_lambda

    # Refit on full training set with the selected lambda
    cat(sprintf("  Refitting on full training set with lambda = %.3f\n", penalty_used))
    model <- SICNN_Net(
      problem_type = "regression",
      sizes        = SIZES,
      input_skip   = TRUE,
      device       = "cpu"
    )

    train_SICNN(
      epochs        = EPOCHS,
      restarts      = 1,
      SICNN         = model,
      lr            = LR,
      train_dl      = loaders$train_loader,
      device        = "cpu",
      scheduler     = "step",
      sch_step_size = SCH_STEP_SIZE,
      n_train       = n_train,
      epsilon_1     = EPSILON_1,
      epsilon_T     = EPSILON_T,
      steps_T       = STEPS_T,
      sic_threshold = SIC_THRESHOLD,
      penalty       = penalty_used
    )
  }

  # Extract and store metrics
  results[[i]] <- extract_metrics(
    model        = model,
    loaders      = loaders,
    n            = exp_i$n,
    snr          = exp_i$snr,
    rep_id       = exp_i$rep,
    method       = method,
    penalty_used = penalty_used,
    p            = p,
    beta_true    = beta_true
  )
}

# ── Results ───────────────────────────────────────────────────────────────────

final_results <- bind_rows(results)

# Summary table grouped by method
summary_table <- final_results %>%
  group_by(method, n, snr) %>%
  summarise(
    mean_penalty   = mean(penalty_used),
    sd_penalty     = sd(penalty_used),
    mean_mse       = mean(test_mse),
    sd_mse         = sd(test_mse),
    mean_coef_err  = mean(coef_error),
    sd_coef_err    = sd(coef_error),
    mean_tpr       = mean(tpr),
    mean_fpr       = mean(fpr),
    mean_active_w  = mean(active_weights),
    sd_active_w    = sd(active_weights),
    .groups        = "drop"
  )

cat("\n\n====== Simulation Summary ======\n")
print(as.data.frame(summary_table))

# Optionally save results
# saveRDS(final_results, "linear_sim_results.rds")
# write.csv(summary_table, "linear_sim_summary.csv", row.names = FALSE)
