library(devtools)
load_all(".")
library(dplyr)
library(tidyr)
library(purrr)

# Function to generate additive model data
generate_additive_data <- function(n, p, snr) {
  # X from Uniform(-1, 1) to cover meaningful ranges for sin, exp, and x^2
  X <- matrix(runif(n * p, 0, 2), ncol = p)

  # Yi = sin(πXi1) + 2.5Xi2^2 - 2.5exp(-2Xi3^2) + εi
  signal <- sin(pi * X[, 1]) + 2.5 * X[, 2]^2 - 2.5 * exp(-2 * X[, 3]^2)

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
# Default BIC penalty for n=800 is log(800) approx 6.68
# We will test a range of penalties to see which recovers the structure best

penalties_list <- c(log(800), 10, 20, 50, 100)
n_reps <- 1
p <- 15

results <- list()

set.seed(42)
torch::torch_manual_seed(42)

# Grid of experiments
experiments <- expand.grid(
  n = ns_list,
  snr = snrs_list,
  penalty = penalties_list,
  rep = 1:n_reps
)

cat("Starting Additive Model simulation with", nrow(experiments), "total runs...\n")

for (i in 1:nrow(experiments)) {
  exp <- experiments[i, ]
  cat(sprintf(
    "Run %d/%d: N=%d, SNR=%d, Penalty=%.2f, Rep=%d\n",
    i, nrow(experiments), exp$n, exp$snr, exp$penalty, exp$rep
  ))

  # Generate data
  data <- generate_additive_data(exp$n, p, exp$snr)

  # Loaders
  loaders <- get_dataloaders(
    data,
    train_proportion = 0.8,
    train_batch_size = min(as.integer(exp$n * 0.8), 200),
    test_batch_size = min(as.integer(exp$n * 0.2), 100),
    standardize = FALSE
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
    epochs = 5000,
    restarts = 1,
    SICNN = model,
    lr = 0.002,
    train_dl = loaders$train_loader,
    device = "cpu",
    scheduler = "step",
    sch_step_size = 1500,
    n_train = exp$n * 0.8,
    epsilon_1 = 1,
    epsilon_T = 1e-5,
    steps_T = 1000,
    sic_threshold = 0.5,
    penalty = exp$penalty
  )

  # Metrics
  val_res <- validate_SICNN(model, num_samples = 1, test_dl = loaders$test_loader, device = "cpu", verbose = FALSE)
  test_mse <- as.numeric(val_res$validation_error_sparse)

  # Feature Selection
  model$compute_paths_input_skip(epsilon = 1e-5, threshold = 0.5)

  selected <- rep(FALSE, p)
  # Check hidden layers
  for (l in model$layers$children) {
    alp <- as.matrix(l$alpha_active_path$cpu())
    in_f <- ncol(alp)
    cov_cols <- if (in_f == p) 1:p else (in_f - p + 1):in_f
    selected <- selected | (colSums(alp[, cov_cols, drop = FALSE]) > 0)
  }
  # Check output layer
  alp_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  in_f <- ncol(alp_out)
  cov_cols <- if (in_f == p) 1:p else (in_f - p + 1):in_f
  selected <- selected | (colSums(alp_out[, cov_cols, drop = FALSE]) > 0)

  # Additive Structure Recovery Metric
  compute_additivity_ratio <- function(model, p) {
    current_masks <- diag(p)
    interaction_count <- 0
    active_count <- 0
    num_h_layers <- length(model$layers$children)

    # Layer 1
    l1 <- model$layers$children$`0`
    alp1 <- as.matrix(l1$alpha_active_path$cpu())
    next_masks <- (alp1 %*% current_masks) > 0

    active_row <- rowSums(alp1) > 0
    interaction_count <- interaction_count + sum((rowSums(next_masks) > 1) & active_row)
    active_count <- active_count + sum(active_row)

    # Subsequent Layers
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
    return(1 - (interaction_count / active_count))
  }

  additivity_ratio <- compute_additivity_ratio(model, p)

  # TPR/FPR
  true_vars <- 1:3
  false_vars <- 4:p

  tpr <- sum(selected[true_vars]) / length(true_vars)
  fpr <- sum(selected[false_vars]) / length(false_vars)

  # Store results
  results[[i]] <- tibble(
    n = exp$n,
    snr = exp$snr,
    penalty = exp$penalty,
    rep = exp$rep,
    test_mse = test_mse,
    tpr = tpr,
    fpr = fpr,
    additivity = additivity_ratio
  )
}

final_results <- bind_rows(results)

# Summary table
summary_table <- final_results %>%
  group_by(n, snr, penalty) %>%
  summarise(
    mean_mse = mean(test_mse),
    mean_tpr = mean(tpr),
    mean_fpr = mean(fpr),
    mean_additivity = mean(additivity),
    .groups = "drop"
  )

print(summary_table)

# Optional: Plot the response for a variable to see if non-linearity was captured
# plot(model, data = loaders$test_loader$dataset$tensors[[1]][1, ], type = "local")
