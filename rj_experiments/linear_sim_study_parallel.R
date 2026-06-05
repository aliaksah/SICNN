library(future)
library(furrr)
library(dplyr)
library(tibble)

# ── Install SICNN so all parallel workers can library(SICNN) ─────────────────
# Workers are separate R processes (PSOCK); devtools::load_all() only affects
# the parent session.  Installing once here makes the package available to all.
message("Installing SICNN for parallel workers (this may take ~30s)...")
devtools::install(
  "C:/Users/Andrew.McInerney/SICNN",
  quiet        = TRUE,
  upgrade      = "never",
  dependencies = FALSE
)
message("Done.")

# ── Simulation settings (shared constants) ────────────────────────────────────
P             <- 15L
BETA_TRUE     <- c(0.6, -0.4, 0.5, rep(0, P - 3))
LAMBDA_GRID   <- exp(seq(log(1), log(100), length.out = 10))
EPOCHS        <- 2000L
LR            <- 0.002
SCH_STEP_SIZE <- 500L
SIZES         <- c(P, 5L, 5L, 1L)
EPSILON_1     <- 1
EPSILON_T     <- 1e-5
STEPS_T       <- 200L
SIC_THRESHOLD <- 0.5
N_WORKERS     <- 11L          
cat("Lambda grid (10 log-spaced values from 1 to 100):\n")
print(round(LAMBDA_GRID, 3))

# ── Experiment grid ───────────────────────────────────────────────────────────
experiments <- expand.grid(
  n      = 2000L,
  snr    = c(3, 5, 10),
  rep    = 1:5,
  method = c("bic", "cv"),
  stringsAsFactors = FALSE
)
cat(sprintf("\nTotal runs: %d  |  Workers: %d  |  Est. wall time: ~35 min\n\n",
            nrow(experiments), N_WORKERS))

# ── Self-contained worker function ────────────────────────────────────────────
# Everything each worker needs is passed as an argument; no globals assumed.

run_one_experiment <- function(
    exp_row,
    p, beta_true, lambda_grid,
    epochs, lr, sch_step_size, sizes,
    epsilon_1, epsilon_T, steps_T, sic_threshold
) {
  # ── Load packages in worker ───────────────────────────────────────────────
  library(SICNN)
  library(torch)
  library(tibble)

  # Limit torch's internal thread pool: with N workers already saturating the
  # CPU, extra threads inside torch cause harmful oversubscription.
  torch::torch_set_num_threads(1L)

  # ── Reproducible seeds per experiment ────────────────────────────────────
  # furrr handles R's RNG; set torch seed separately.
  torch::torch_manual_seed(exp_row$rep * 1000L + exp_row$snr * 10L +
                              as.integer(exp_row$method == "cv"))

  # ── Helper: generate data ─────────────────────────────────────────────────
  make_data <- function(n, p, snr, beta_true) {
    X        <- matrix(rnorm(n * p), ncol = p)
    signal   <- X %*% beta_true
    var_sig  <- var(as.numeric(signal))
    noise    <- rnorm(n, sd = sqrt(var_sig / snr))
    sim_df   <- as.data.frame(X)
    colnames(sim_df) <- paste0("x", seq_len(p))
    sim_df[["y"]] <- as.numeric(signal) + noise
    sim_df
  }

  # ── Helper: build torch dataloaders directly (bypasses get_dataloaders
  #    which requires train_proportion < 1 for both splits).
  make_loader <- function(sim_df, p, batch_size, shuffle = TRUE) {
    xmat <- as.matrix(sim_df[, seq_len(p)])
    yvec <- as.numeric(sim_df[["y"]])
    ds   <- torch::tensor_dataset(torch::torch_tensor(xmat),
                                  torch::torch_tensor(yvec))
    torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)
  }

  # ── Helper: train one model ───────────────────────────────────────────────
  fit_model <- function(train_loader, n_train, penalty_val) {
    mdl <- SICNN_Net(
      problem_type = "regression",
      sizes        = sizes,
      input_skip   = TRUE,
      device       = "cpu"
    )
    train_SICNN(
      epochs        = epochs,
      restarts      = 1L,
      SICNN         = mdl,
      lr            = lr,
      train_dl      = train_loader,
      device        = "cpu",
      scheduler     = "step",
      sch_step_size = sch_step_size,
      n_train       = n_train,
      epsilon_1     = epsilon_1,
      epsilon_T     = epsilon_T,
      steps_T       = steps_T,
      sic_threshold = sic_threshold,
      penalty       = penalty_val   # NULL → BIC default
    )
    mdl
  }

  # ── Helper: CV lambda selection (held-out validation fold) ────────────────
  select_lambda_cv <- function(sim_df, p, lambda_grid) {
    n_total     <- nrow(sim_df)
    val_idx     <- sample(n_total, size = round(n_total * 0.10))
    cv_train_df <- sim_df[-val_idx, ]
    cv_val_df   <- sim_df[ val_idx, ]
    n_cv_train  <- nrow(cv_train_df)

    cv_train_loader <- make_loader(cv_train_df, p,
                                   batch_size = min(n_cv_train, 200L),
                                   shuffle    = TRUE)
    cv_val_loader   <- make_loader(cv_val_df, p,
                                   batch_size = min(nrow(cv_val_df), 100L),
                                   shuffle    = FALSE)

    cv_rmse <- numeric(length(lambda_grid))

    for (j in seq_along(lambda_grid)) {
      cv_mdl <- fit_model(cv_train_loader, n_cv_train, lambda_grid[j])
      cv_mdl$eval()

      # Activate sparse paths at final epsilon
      cv_mdl$compute_paths_input_skip(epsilon = epsilon_T,
                                      threshold = sic_threshold)

      sq_err <- numeric(0)
      torch::with_no_grad({
        coro::loop(for (b in cv_val_loader) {
          pred   <- cv_mdl(b[[1]], sparse = TRUE)$squeeze()
          target <- b[[2]]
          sq_err <- c(sq_err, as.numeric((pred - target)^2))
        })
      })
      cv_rmse[j] <- sqrt(mean(sq_err))
    }

    best_idx <- which.min(cv_rmse)
    list(
      best_lambda = lambda_grid[best_idx],
      best_rmse   = cv_rmse[best_idx],
      all_lambdas = lambda_grid,
      all_rmse    = cv_rmse
    )
  }

  # ── Helper: extract all metrics from a fitted model ───────────────────────
  extract_metrics <- function(model, test_loader, p, beta_true,
                              n, snr, rep_id, method, penalty_used) {
    # Activate sparse paths
    model$compute_paths_input_skip(epsilon = epsilon_T, threshold = sic_threshold)

    # Test RMSE (sparse model)
    model$eval()
    sq_err <- numeric(0)
    torch::with_no_grad({
      coro::loop(for (b in test_loader) {
        pred   <- model(b[[1]], sparse = TRUE)$squeeze()
        target <- b[[2]]
        sq_err <- c(sq_err, as.numeric((pred - target)^2))
      })
    })
    test_mse <- mean(sq_err)   # MSE (not RMSE)

    # Coefficient recovery
    cf        <- coef(model, dataset = test_loader, num_data = 10L, num_samples = 1L)
    beta_hat  <- cf$mean
    coef_err  <- sqrt(sum((beta_hat - beta_true)^2))

    # Feature selection via active paths
    selected <- rep(FALSE, p)
    for (l in model$layers$children) {
      alp      <- as.matrix(l$alpha_active_path$cpu())
      in_f     <- ncol(alp)
      cols     <- if (in_f == p) seq_len(p) else (in_f - p + 1L):in_f
      selected <- selected | (colSums(alp[, cols, drop = FALSE]) > 0)
    }
    alp_out  <- as.matrix(model$out_layer$alpha_active_path$cpu())
    in_f     <- ncol(alp_out)
    cols     <- if (in_f == p) seq_len(p) else (in_f - p + 1L):in_f
    selected <- selected | (colSums(alp_out[, cols, drop = FALSE]) > 0)

    tpr <- sum(selected[1:3])  / 3
    fpr <- sum(selected[4:p])  / (p - 3)

    sic_counts     <- model$sic_weight_counts(epsilon = epsilon_T,
                                              threshold = sic_threshold,
                                              active_paths = TRUE)
    active_weights <- as.numeric(sic_counts["active"])

    tibble(
      n              = n,
      snr            = snr,
      rep            = rep_id,
      method         = method,
      penalty_used   = penalty_used,
      test_mse       = test_mse,
      coef_error     = coef_err,
      tpr            = tpr,
      fpr            = fpr,
      active_weights = active_weights
    )
  }

  # ── Main experiment logic ─────────────────────────────────────────────────
  sim_df  <- make_data(exp_row$n, p, exp_row$snr, beta_true)
  n_train <- as.integer(exp_row$n * 0.8)
  n_test  <- exp_row$n - n_train

  # Split data into train / test (reproducible via R's RNG managed by furrr)
  train_idx   <- sample(nrow(sim_df), n_train)
  train_df    <- sim_df[ train_idx, ]
  test_df     <- sim_df[-train_idx, ]

  train_loader <- make_loader(train_df, p,
                              batch_size = min(n_train, 200L), shuffle = TRUE)
  test_loader  <- make_loader(test_df,  p,
                              batch_size = min(n_test,  100L), shuffle = FALSE)

  if (exp_row$method == "bic") {
    # ── BIC: penalty = log(n_train) (default) ──────────────────────────────
    penalty_used <- log(n_train)
    model <- fit_model(train_loader, n_train, penalty_val = NULL)

  } else {
    # ── CV: select penalty from lambda_grid on a held-out validation fold ──
    cv_out       <- select_lambda_cv(sim_df, p, lambda_grid)
    penalty_used <- cv_out$best_lambda

    # Refit on the full training split with the selected lambda
    model <- fit_model(train_loader, n_train, penalty_val = penalty_used)
  }

  extract_metrics(
    model        = model,
    test_loader  = test_loader,
    p            = p,
    beta_true    = beta_true,
    n            = exp_row$n,
    snr          = exp_row$snr,
    rep_id       = exp_row$rep,
    method       = exp_row$method,
    penalty_used = penalty_used
  )
}

# ── Run in parallel ───────────────────────────────────────────────────────────
plan(multisession, workers = N_WORKERS)

cat(sprintf("Launching %d experiments across %d workers...\n",
            nrow(experiments), N_WORKERS))
t_start <- proc.time()

final_results <- future_map_dfr(
  seq_len(nrow(experiments)),
  function(i) {
    run_one_experiment(
      exp_row       = experiments[i, ],
      p             = P,
      beta_true     = BETA_TRUE,
      lambda_grid   = LAMBDA_GRID,
      epochs        = EPOCHS,
      lr            = LR,
      sch_step_size = SCH_STEP_SIZE,
      sizes         = SIZES,
      epsilon_1     = EPSILON_1,
      epsilon_T     = EPSILON_T,
      steps_T       = STEPS_T,
      sic_threshold = SIC_THRESHOLD
    )
  },
  .options = furrr_options(seed = TRUE)   # reproducible parallel RNG
)

elapsed <- (proc.time() - t_start)[[3]]
cat(sprintf("\nSimulation complete in %.1f min.\n", elapsed / 60))

plan(sequential)   # release workers

# ── Save results ──────────────────────────────────────────────────────────────
out_path <- "C:/Users/Andrew.McInerney/SICNN/rj_experiments/linear_sim_results.rds"
saveRDS(final_results, out_path)
cat(sprintf("Raw results saved to: %s\n", out_path))

# ── Summary table ─────────────────────────────────────────────────────────────
summary_table <- final_results |>
  group_by(method, snr) |>
  summarise(
    mean_penalty  = mean(penalty_used),
    sd_penalty    = sd(penalty_used),
    mean_mse      = mean(test_mse),
    sd_mse        = sd(test_mse),
    mean_coef_err = mean(coef_error),
    sd_coef_err   = sd(coef_error),
    mean_tpr      = mean(tpr),
    mean_fpr      = mean(fpr),
    mean_active_w = mean(active_weights),
    .groups       = "drop"
  )

cat("\n====== Simulation Summary ======\n")
print(as.data.frame(summary_table))
