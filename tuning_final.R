library(devtools)
load_all(".")

i <- 1000
j <- 15
set.seed(42)
torch::torch_manual_seed(42)
X <- matrix(rnorm(i * j, mean = 0, sd = 1), ncol = j)
y_base <- 0.6 * X[, 1] - 0.4 * X[, 2] + 0.5 * X[, 3] + rnorm(n = i, sd = 0.1)
sim_data <- as.data.frame(X)
sim_data <- cbind(sim_data, y_base)

loaders <- get_dataloaders(sim_data, train_proportion = 0.9, train_batch_size = 450, 
                           test_batch_size = 100, standardize = FALSE)
train_loader <- loaders$train_loader

penalties <- c(30,35,40,45,50,60,70,80)
results <- list()

for (p in penalties) {
  cat("\n\n--- Testing Penalty:", p, "---\n")
  model <- SICNN_Net(problem_type = "regression", sizes = c(j, 5, 5, 1), 
                     input_skip = TRUE, device = "cpu")
  
  res <- train_SICNN(
    epochs = 2000, restarts = 1, SICNN = model, lr = 0.002, 
    train_dl = train_loader, device = "cpu",
    scheduler = "step", sch_step_size = 500, n_train = i,
    epsilon_1 = 1, epsilon_T = 1e-5, steps_T = 200,
    sic_threshold = 0.5, penalty = p
  )
  
  counts <- model$sic_weight_counts(epsilon = 1e-5, threshold = 0.5, active_paths = TRUE)
  cf <- coef(model, dataset = train_loader, inds = 1, num_data = 1)
  
  results[[as.character(p)]] <- list(counts = counts, coefs = cf)
  cat("Active Weights:", counts["active"], "\n")
  cat("Signals Detected (non-zero):\n")
  print(cf[cf$mean != 0, , drop = FALSE])
}

print(results)
