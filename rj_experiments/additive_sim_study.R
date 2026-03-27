
library(dplyr)
library(tidyr)
library(purrr)

# Function to generate additive model data
generate_additive_data <- function(n, p, snr) {
  # X from Uniform(-1, 1) to cover meaningful ranges for sin, exp, and x^2
  X <- matrix(runif(n * p, -1, 1), ncol = p)
  
  # Yi = 1.5sin(πXi1) + 2Xi2^2 - 1.5exp(-2Xi3^2) + εi
  signal <- 1.5 * sin(pi * X[, 1]) + 2 * X[, 2]^2 - 1.5 * exp(-2 * X[, 3]^2)
  
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

results <- list()

set.seed(42)
torch::torch_manual_seed(42)

# Grid of experiments
experiments <- expand.grid(
  n = ns_list,
  snr = snrs_list,
  rep = 1:n_reps
)

cat("Starting Additive Model simulation with", nrow(experiments), "total runs...\n")

for (i in 1:nrow(experiments)) {
  exp <- experiments[i, ]
  cat(sprintf("Run %d/%d: N=%d, SNR=%d, Rep=%d\n", i, nrow(experiments), exp$n, exp$snr, exp$rep))

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
  # For additive models, we need enough hidden neurons to capture the non-linearity. 
  # Hidden layers (10, 10) should be sufficient for these functions.
  model <- SICNN_Net(
    problem_type = "regression",
    sizes = c(p, 10, 10, 1),
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
    sch_step_size = 500,
    n_train = exp$n * 0.8,
    epsilon_1 = 1,
    epsilon_T = 1e-5,
    steps_T = 500,
    sic_threshold = 0.5,
    penalty = NULL
  )

  # Metrics
  # 1. Test MSE (Validation)
  val_res <- validate_SICNN(model, num_samples = 1, test_dl = loaders$test_loader, device = "cpu", verbose = FALSE)
  # Extract numeric MSE from the sparse model prediction (frequentist point estimate)
  test_mse <- as.numeric(val_res$validation_error_sparse)

  # 2. Feature Selection (Frequentist Selection via Active Paths)
  model$compute_paths_input_skip(epsilon = 1e-5, threshold = 0.5)

  # Determine if each input xj is part of any active path to the output
  selected <- rep(FALSE, p)
  for (l in model$layers$children) {
    alp <- as.matrix(l$alpha_active_path$cpu())
    in_f <- ncol(alp)
    cov_cols <- if (in_f == p) 1:p else (in_f - p + 1):in_f
    selected <- selected | (colSums(alp[, cov_cols, drop = FALSE]) > 0)
  }
  alp_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  in_f <- ncol(alp_out)
  cov_cols <- if (in_f == p) 1:p else (in_f - p + 1):in_f
  selected <- selected | (colSums(alp_out[, cov_cols, drop = FALSE]) > 0)

  # 3. Additive Structure Recovery Metric
  # We check if hidden nodes aggregate information from multiple inputs.
  compute_additivity_ratio <- function(model, p) {
    current_masks <- diag(p) # Each input reaches itself
    interaction_count <- 0
    active_count <- 0
    num_h_layers <- length(model$layers$children)
    
    # Layer 1
    l1 <- model$layers$children$`0`
    alp1 <- as.matrix(l1$alpha_active_path$cpu())
    next_masks <- (alp1 %*% current_masks) > 0 # (out, p)
    
    active_row <- rowSums(alp1) > 0
    interaction_count <- interaction_count + sum((rowSums(next_masks) > 1) & active_row)
    active_count <- active_count + sum(active_row)
    
    # Subsequent Layers
    if (num_h_layers > 1) {
      for (i in 2:num_h_layers) {
        l <- model$layers$children[[i-1]]
        alp <- as.matrix(l$alpha_active_path$cpu())
        in_masks <- rbind(next_masks, diag(p)) # hidden + inputs
        next_masks <- (alp %*% in_masks) > 0
        
        active_row <- rowSums(alp) > 0
        interaction_count <- interaction_count + sum((rowSums(next_masks) > 1) & active_row)
        active_count <- active_count + sum(active_row)
      }
    }
    
    if (active_count == 0) return(1)
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
  group_by(n, snr) %>%
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
