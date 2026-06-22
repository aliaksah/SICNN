library(devtools)
load_all(".")
library(torch)
library(dplyr)
library(tibble)

# Generate additive regression data:
# y = sin(pi * x1) + 2.5 * x2^2 - 2.5 * exp(-2 * x3^2) + noise
generate_additive_data <- function(n, p, snr) {
  X <- matrix(runif(n * p, 0, 2), ncol = p)
  signal <- sin(pi * X[, 1]) + 2.5 * X[, 2]^2 - 2.5 * exp(-2 * X[, 3]^2)
  var_signal <- var(signal)
  noise <- rnorm(n, sd = sqrt(var_signal / snr))

  data <- as.data.frame(X)
  colnames(data) <- paste0("x", seq_len(p))
  data$y <- as.numeric(signal + noise)
  data
}

make_loader <- function(sim_df, p, batch_size, shuffle = TRUE) {
  xmat <- as.matrix(sim_df[, seq_len(p)])
  yvec <- as.numeric(sim_df[["y"]])
  ds <- torch::tensor_dataset(
    torch::torch_tensor(xmat),
    torch::torch_tensor(yvec)
  )
  torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)
}

fit_model <- function(train_loader, n_train, penalty_val,
                      epochs, lr, sch_step_size, sizes,
                      epsilon_1, epsilon_T, steps_T, sic_threshold) {
  model <- SICNN_Net(
    problem_type = "regression",
    sizes = sizes,
    input_skip = TRUE,
    device = "cpu"
  )

  train_SICNN(
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
    penalty = penalty_val
  )

  model
}

# Pick lambda with a single held-out validation fold, matching the linear study.
cv_lambda <- function(data, lambda_grid, p,
                      epochs, lr, sch_step_size, sizes,
                      epsilon_1, epsilon_T, steps_T, sic_threshold) {
  n_total <- nrow(data)
  val_idx <- sample(n_total, size = round(n_total * 0.10))
  cv_train_df <- data[-val_idx, ]
  cv_val_df <- data[val_idx, ]
  n_cv_train <- nrow(cv_train_df)

  cv_train_loader <- make_loader(
    cv_train_df, p,
    batch_size = min(n_cv_train, 200L),
    shuffle = TRUE
  )
  cv_val_loader <- make_loader(
    cv_val_df, p,
    batch_size = min(nrow(cv_val_df), 100L),
    shuffle = FALSE
  )

  cv_rmse <- numeric(length(lambda_grid))

  for (j in seq_along(lambda_grid)) {
    lam <- lambda_grid[j]
    cat(sprintf("  [CV] lambda %d/%d = %.3f\n", j, length(lambda_grid), lam))

    cv_model <- fit_model(
      cv_train_loader, n_cv_train, lam,
      epochs, lr, sch_step_size, sizes,
      epsilon_1, epsilon_T, steps_T, sic_threshold
    )

    cv_model$compute_paths_input_skip(
      epsilon = epsilon_T,
      threshold = sic_threshold
    )
    cv_model$eval()

    sq_err <- numeric(0)
    torch::with_no_grad({
      coro::loop(for (b in cv_val_loader) {
        pred <- cv_model(b[[1]], sparse = TRUE)$squeeze()
        target <- b[[2]]
        sq_err <- c(sq_err, as.numeric((pred - target)^2))
      })
    })
    cv_rmse[j] <- sqrt(mean(sq_err))
  }

  best_idx <- which.min(cv_rmse)
  cat(sprintf(
    "  [CV] best lambda = %.3f (RMSE = %.4f)\n",
    lambda_grid[best_idx], cv_rmse[best_idx]
  ))

  list(
    best_lambda = lambda_grid[best_idx],
    best_rmse = cv_rmse[best_idx],
    all_lambdas = lambda_grid,
    all_rmse = cv_rmse
  )
}

select_active_features <- function(model, p, epsilon_T, sic_threshold) {
  model$compute_paths_input_skip(epsilon = epsilon_T, threshold = sic_threshold)

  selected <- rep(FALSE, p)

  for (l in model$layers$children) {
    alp <- as.matrix(l$alpha_active_path$cpu())
    in_f <- ncol(alp)
    cov_cols <- if (in_f == p) seq_len(p) else (in_f - p + 1L):in_f
    selected <- selected | (colSums(alp[, cov_cols, drop = FALSE]) > 0)
  }

  alp_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  in_f <- ncol(alp_out)
  cov_cols <- if (in_f == p) seq_len(p) else (in_f - p + 1L):in_f
  selected <- selected | (colSums(alp_out[, cov_cols, drop = FALSE]) > 0)

  selected
}

compute_additivity_ratio <- function(model, p) {
  current_masks <- diag(p)
  interaction_count <- 0
  active_count <- 0
  num_h_layers <- length(model$layers$children)

  l1 <- model$layers$children$`0`
  alp1 <- as.matrix(l1$alpha_active_path$cpu())
  next_masks <- (alp1 %*% current_masks) > 0

  active_row <- rowSums(alp1) > 0
  interaction_count <- interaction_count + sum((rowSums(next_masks) > 1) & active_row)
  active_count <- active_count + sum(active_row)

  if (num_h_layers > 1) {
    for (idx in 2:num_h_layers) {
      l <- model$layers$children[[idx]]
      alp <- as.matrix(l$alpha_active_path$cpu())
      in_masks <- rbind(next_masks, diag(p))
      next_masks <- (alp %*% in_masks) > 0

      active_row <- rowSums(alp) > 0
      interaction_count <- interaction_count + sum((rowSums(next_masks) > 1) & active_row)
      active_count <- active_count + sum(active_row)
    }
  }

  if (active_count == 0) {
    return(1)
  }
  1 - (interaction_count / active_count)
}

extract_metrics <- function(model, test_loader, p, n, snr, rep_id,
                            method, penalty_used, epsilon_T, sic_threshold) {
  selected <- select_active_features(model, p, epsilon_T, sic_threshold)
  additivity_ratio <- compute_additivity_ratio(model, p)

  model$eval()
  sq_err <- numeric(0)
  torch::with_no_grad({
    coro::loop(for (b in test_loader) {
      pred <- model(b[[1]], sparse = TRUE)$squeeze()
      target <- b[[2]]
      sq_err <- c(sq_err, as.numeric((pred - target)^2))
    })
  })

  true_vars <- 1:3
  false_vars <- 4:p
  sic_counts <- model$sic_weight_counts(
    epsilon = epsilon_T,
    threshold = sic_threshold,
    active_paths = TRUE
  )

  tibble(
    n = n,
    snr = snr,
    rep = rep_id,
    method = method,
    penalty_used = penalty_used,
    test_mse = mean(sq_err),
    tpr = sum(selected[true_vars]) / length(true_vars),
    fpr = sum(selected[false_vars]) / length(false_vars),
    additivity = additivity_ratio,
    active_weights = as.numeric(sic_counts["active"])
  )
}

# Simulation settings
ns_list <- c(2000L)
snrs_list <- c(3)
n_reps <- 5L
p <- 15L

lambda_grid <- exp(seq(log(1), log(100), length.out = 10))
cat("Lambda grid:\n")
print(round(lambda_grid, 3))

EPOCHS <- 5000L
LR <- 0.002
SCH_STEP_SIZE <- 1500L
SIZES <- c(p, 5L, 5L, 1L)
EPSILON_1 <- 10
EPSILON_T <- 1e-5
STEPS_T <- 100L
SIC_THRESHOLD <- 0.5

set.seed(42)
torch::torch_manual_seed(42)

experiments <- expand.grid(
  n = ns_list,
  snr = snrs_list,
  rep = seq_len(n_reps),
  method = c("bic", "cv"),
  stringsAsFactors = FALSE
)

cat(sprintf(
  "Starting additive simulation with %d total runs.\n",
  nrow(experiments)
))

results <- vector("list", nrow(experiments))

for (i in seq_len(nrow(experiments))) {
  exp_i <- experiments[i, ]
  method <- exp_i$method

  cat(sprintf(
    "\nRun %d/%d: N=%d, SNR=%d, Rep=%d, Method=%s\n",
    i, nrow(experiments), exp_i$n, exp_i$snr, exp_i$rep, method
  ))

  data <- generate_additive_data(exp_i$n, p, exp_i$snr)
  n_train <- as.integer(exp_i$n * 0.8)
  n_test <- exp_i$n - n_train

  train_idx <- sample(nrow(data), n_train)
  train_df <- data[train_idx, ]
  test_df <- data[-train_idx, ]

  train_loader <- make_loader(
    train_df, p,
    batch_size = min(n_train, 200L),
    shuffle = TRUE
  )
  test_loader <- make_loader(
    test_df, p,
    batch_size = min(n_test, 100L),
    shuffle = FALSE
  )

  if (method == "bic") {
    penalty_used <- log(n_train)
    model <- fit_model(
      train_loader, n_train, penalty_val = NULL,
      EPOCHS, LR, SCH_STEP_SIZE, SIZES,
      EPSILON_1, EPSILON_T, STEPS_T, SIC_THRESHOLD
    )
  } else if (method == "cv") {
    cat("  Running CV to select lambda...\n")
    cv_out <- cv_lambda(
      data, lambda_grid, p,
      EPOCHS, LR, SCH_STEP_SIZE, SIZES,
      EPSILON_1, EPSILON_T, STEPS_T, SIC_THRESHOLD
    )
    penalty_used <- cv_out$best_lambda

    cat(sprintf("  Refitting on full training set with lambda = %.3f\n", penalty_used))
    model <- fit_model(
      train_loader, n_train, penalty_used,
      EPOCHS, LR, SCH_STEP_SIZE, SIZES,
      EPSILON_1, EPSILON_T, STEPS_T, SIC_THRESHOLD
    )
  }

  results[[i]] <- extract_metrics(
    model = model,
    test_loader = test_loader,
    p = p,
    n = exp_i$n,
    snr = exp_i$snr,
    rep_id = exp_i$rep,
    method = method,
    penalty_used = penalty_used,
    epsilon_T = EPSILON_T,
    sic_threshold = SIC_THRESHOLD
  )
}

final_results <- bind_rows(results)

summary_table <- final_results |>
  group_by(method, n, snr) |>
  summarise(
    mean_penalty = mean(penalty_used),
    sd_penalty = sd(penalty_used),
    mean_mse = mean(test_mse),
    sd_mse = sd(test_mse),
    mean_tpr = mean(tpr),
    mean_fpr = mean(fpr),
    mean_additivity = mean(additivity),
    sd_additivity = sd(additivity),
    mean_active_w = mean(active_weights),
    sd_active_w = sd(active_weights),
    .groups = "drop"
  )

cat("\n====== Additive Simulation Summary ======\n")
print(as.data.frame(summary_table))

# Optionally save results
# saveRDS(final_results, "rj_experiments/additive_sim_results.rds")
# write.csv(summary_table, "rj_experiments/additive_sim_summary.csv", row.names = FALSE)
