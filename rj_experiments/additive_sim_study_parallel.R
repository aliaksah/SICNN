library(future)
library(furrr)
library(dplyr)
library(tibble)

# Workers are separate R processes (PSOCK); devtools::load_all() only affects
# the parent session. Installing once here makes the package available to all.
message("Installing SICNN for parallel workers (this may take ~30s)...")
devtools::install(
  "C:/Users/Andrew.McInerney/SICNN",
  quiet = TRUE,
  upgrade = "never",
  dependencies = FALSE
)
message("Done.")

# Shared simulation settings
P <- 15L
LAMBDA_GRID <- exp(seq(log(1), log(100), length.out = 10))
EPOCHS <- 5000L
LR <- 0.002
SCH_STEP_SIZE <- 1500L
SIZES <- c(P, 5L, 5L, 1L)
EPSILON_1 <- 10
EPSILON_T <- 1e-5
STEPS_T <- 100L
SIC_THRESHOLD <- 0.5
N_WORKERS <- 11L

cat("Lambda grid (10 log-spaced values from 1 to 100):\n")
print(round(LAMBDA_GRID, 3))

experiments <- expand.grid(
  n = 2000L,
  snr = c(3, 5, 10),
  rep = 1:10,
  method = c("bic", "cv"),
  stringsAsFactors = FALSE
)

cat(sprintf(
  "\nTotal runs: %d  |  Workers: %d\n\n",
  nrow(experiments), N_WORKERS
))

run_one_experiment <- function(
    exp_row,
    p, lambda_grid,
    epochs, lr, sch_step_size, sizes,
    epsilon_1, epsilon_T, steps_T, sic_threshold
) {
  library(SICNN)
  library(torch)
  library(tibble)

  torch::torch_set_num_threads(1L)
  torch::torch_manual_seed(
    exp_row$rep * 1000L + exp_row$snr * 10L +
      as.integer(exp_row$method == "cv")
  )

  make_data <- function(n, p, snr) {
    X <- matrix(runif(n * p, 0, 2), ncol = p)
    signal <- sin(pi * X[, 1]) + 2.5 * X[, 2]^2 - 2.5 * exp(-2 * X[, 3]^2)
    var_signal <- var(signal)
    noise <- rnorm(n, sd = sqrt(var_signal / snr))

    sim_df <- as.data.frame(X)
    colnames(sim_df) <- paste0("x", seq_len(p))
    sim_df[["y"]] <- as.numeric(signal + noise)
    sim_df
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

  fit_model <- function(train_loader, n_train, penalty_val) {
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

  select_lambda_cv <- function(sim_df, p, lambda_grid) {
    n_total <- nrow(sim_df)
    val_idx <- sample(n_total, size = round(n_total * 0.10))
    cv_train_df <- sim_df[-val_idx, ]
    cv_val_df <- sim_df[val_idx, ]
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
      cv_model <- fit_model(cv_train_loader, n_cv_train, lambda_grid[j])
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
    list(
      best_lambda = lambda_grid[best_idx],
      best_rmse = cv_rmse[best_idx],
      all_lambdas = lambda_grid,
      all_rmse = cv_rmse
    )
  }

  select_active_features <- function(model, p) {
    model$compute_paths_input_skip(epsilon = epsilon_T, threshold = sic_threshold)

    selected <- rep(FALSE, p)

    for (l in model$layers$children) {
      alp <- as.matrix(l$alpha_active_path$cpu())
      in_f <- ncol(alp)
      cols <- if (in_f == p) seq_len(p) else (in_f - p + 1L):in_f
      selected <- selected | (colSums(alp[, cols, drop = FALSE]) > 0)
    }

    alp_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
    in_f <- ncol(alp_out)
    cols <- if (in_f == p) seq_len(p) else (in_f - p + 1L):in_f
    selected <- selected | (colSums(alp_out[, cols, drop = FALSE]) > 0)

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

  extract_metrics <- function(model, test_loader, p,
                              n, snr, rep_id, method, penalty_used) {
    selected <- select_active_features(model, p)
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
      tpr = sum(selected[1:3]) / 3,
      fpr = sum(selected[4:p]) / (p - 3),
      additivity = additivity_ratio,
      active_weights = as.numeric(sic_counts["active"])
    )
  }

  sim_df <- make_data(exp_row$n, p, exp_row$snr)
  n_train <- as.integer(exp_row$n * 0.8)
  n_test <- exp_row$n - n_train

  train_idx <- sample(nrow(sim_df), n_train)
  train_df <- sim_df[train_idx, ]
  test_df <- sim_df[-train_idx, ]

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

  if (exp_row$method == "bic") {
    penalty_used <- log(n_train)
    model <- fit_model(train_loader, n_train, penalty_val = NULL)
  } else {
    cv_out <- select_lambda_cv(sim_df, p, lambda_grid)
    penalty_used <- cv_out$best_lambda
    model <- fit_model(train_loader, n_train, penalty_val = penalty_used)
  }

  extract_metrics(
    model = model,
    test_loader = test_loader,
    p = p,
    n = exp_row$n,
    snr = exp_row$snr,
    rep_id = exp_row$rep,
    method = exp_row$method,
    penalty_used = penalty_used
  )
}

plan(multisession, workers = N_WORKERS)

cat(sprintf(
  "Launching %d additive experiments across %d workers...\n",
  nrow(experiments), N_WORKERS
))
t_start <- proc.time()

final_results <- future_map_dfr(
  seq_len(nrow(experiments)),
  function(i) {
    run_one_experiment(
      exp_row = experiments[i, ],
      p = P,
      lambda_grid = LAMBDA_GRID,
      epochs = EPOCHS,
      lr = LR,
      sch_step_size = SCH_STEP_SIZE,
      sizes = SIZES,
      epsilon_1 = EPSILON_1,
      epsilon_T = EPSILON_T,
      steps_T = STEPS_T,
      sic_threshold = SIC_THRESHOLD
    )
  },
  .options = furrr_options(seed = TRUE)
)

elapsed <- (proc.time() - t_start)[[3]]
cat(sprintf("\nSimulation complete in %.1f min.\n", elapsed / 60))

plan(sequential)

out_path <- "C:/Users/Andrew.McInerney/SICNN/rj_experiments/additive_sim_results.rds"
saveRDS(final_results, out_path)
cat(sprintf("Raw results saved to: %s\n", out_path))

summary_table <- final_results |>
  group_by(method, snr) |>
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
