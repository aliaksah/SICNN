library(devtools)
load_all(".")

set.seed(42)
torch::torch_manual_seed(42)

# Generate data
generate_additive_data <- function(n, p, snr) {
  X <- matrix(runif(n * p, -1, 1), ncol = p)
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

data <- generate_additive_data(1000, 15, 20)
loaders <- get_dataloaders(data, train_proportion = 0.8, train_batch_size = 800, test_batch_size = 200, standardize = TRUE)

# Unpenalized mode test
model_no_pen <- SICNN_Net("regression", sizes = c(15, 20, 20, 1), input_skip = TRUE, device="cpu")
train_results_unpen <- train_SICNN(
  epochs = 2000, restarts = 2, SICNN = model_no_pen, lr = 0.05, train_dl = loaders$train_loader, device = "cpu",
  scheduler = "step", sch_step_size = 400, n_train = 800, penalty = 1e-8
)

cat("Final Unpenalized R2: ", tail(train_results_unpen$accs, 1), "\n")
