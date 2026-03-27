
library(dplyr)
library(tidyr)
library(purrr)

# Function to generate linear data
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

# Simulation settings
ns_list <- c(1000)
snrs_list <- c(10)
n_reps <- 2
p <- 15
beta_true <- c(0.6, -0.4, 0.5, rep(0, p - 3))

results <- list()

set.seed(42)
torch::torch_manual_seed(42)

# Grid of experiments
experiments <- expand.grid(
  n = ns_list,
  snr = snrs_list,
  rep = 1:n_reps
)

cat("Starting simulation with", nrow(experiments), "total runs...\n")

for (i in 1:nrow(experiments)) {
  exp <- experiments[i, ]
  cat(sprintf("Run %d/%d: N=%d, SNR=%d, Rep=%d\n", i, nrow(experiments), exp$n, exp$snr, exp$rep))

  # Generate data
  data <- generate_linear_data(exp$n, p, exp$snr, beta_true)

  # Loaders
  loaders <- get_dataloaders(
    data,
    train_proportion = 0.8,
    train_batch_size = min(as.integer(exp$n * 0.8), 200),
    test_batch_size = min(as.integer(exp$n * 0.2), 100),
    standardize = FALSE # Already generated with N(0, 1)
  )

  # Define Model
  model <- SICNN_Net(
    problem_type = "regression",
    sizes = c(p, 5, 5, 1),
    input_skip = TRUE,
    device = "cpu"
  )

  # Train
  train_results <- train_SICNN(
    epochs = 2000,
    restarts = 1,
    SICNN = model,
    lr = 0.002,
    train_dl = loaders$train_loader,
    device = "cpu",
    scheduler = "step",
    sch_step_size = 500,
    n_train = exp$n * 0.8,
    epsilon_1 = 1,
    epsilon_T = 1e-5,
    steps_T = 200,
    sic_threshold = 0.5,
    penalty = 80
  )

  # Metrics
  # 1. Test MSE (Validation)
  val_res <- validate_SICNN(model, num_samples = 1, test_dl = loaders$test_loader, device = "cpu", verbose = FALSE)
  # Extract numeric MSE from the sparse model prediction (frequentist point estimate)
  test_mse <- as.numeric(val_res$validation_error_sparse)

  # 2. Coefficient recovery
  # Using num_samples = 1 for the frequentist point estimate (active paths)
  cf <- coef(model, dataset = loaders$test_loader, num_data = 10, num_samples = 1)
  beta_hat <- cf$mean

  coef_error <- sqrt(sum((beta_hat - beta_true)^2))

  # 3. Feature Selection (Frequentist Selection via Active Paths)
  model$compute_paths_input_skip(epsilon = 1e-5, threshold = 0.5)

  # Determine if each input xj is part of any active path to the output
  selected <- rep(FALSE, p)

  # Path through hidden layers and Skip connections
  for (l in model$layers$children) {
    alp <- as.matrix(l$alpha_active_path$cpu())
    in_f <- ncol(alp)
    cov_cols <- if (in_f == p) 1:p else (in_f - p + 1):in_f
    selected <- selected | (colSums(alp[, cov_cols, drop = FALSE]) > 0)
  }
  # Final layer skip connections
  alp_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  in_f <- ncol(alp_out)
  cov_cols <- if (in_f == p) 1:p else (in_f - p + 1):in_f
  selected <- selected | (colSums(alp_out[, cov_cols, drop = FALSE]) > 0)

  # TPR and FPR
  true_vars <- 1:3
  false_vars <- 4:p

  tpr <- sum(selected[true_vars]) / length(true_vars)
  fpr <- sum(selected[false_vars]) / length(false_vars)

  # Store results
  results[[i]] <- tibble(
    n = exp$n,
    snr = exp$snr,
    rep = exp$rep,
    test_mse = test_mse,
    coef_error = coef_error,
    tpr = tpr,
    fpr = fpr
  )
}

final_results <- bind_rows(results)

# Summary table
summary_table <- final_results %>%
  group_by(n, snr) %>%
  summarise(
    mean_mse = mean(test_mse),
    sd_mse = sd(test_mse),
    mean_coef_err = mean(coef_error),
    mean_tpr = mean(tpr),
    mean_fpr = mean(fpr),
    .groups = "drop"
  )

print(summary_table)

# Plotting the recovery for the last run if you want to see it visually
# plot(model)
