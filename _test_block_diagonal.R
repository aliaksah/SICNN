devtools::load_all("c:/Users/Andrew.McInerney/SICNN", quiet = TRUE)

set.seed(42)
torch::torch_manual_seed(42)
i <- 200; j <- 15
X <- matrix(rnorm(i * j), ncol = j)
y_base <- 0.6 * X[, 1] - 0.4 * X[, 2] + 0.5 * X[, 3] + rnorm(i, sd = 0.1)
sim_data <- as.data.frame(cbind(X, y_base))

loaders <- get_dataloaders(sim_data, train_proportion = 0.9,
                           train_batch_size = 90, test_batch_size = 20,
                           standardize = FALSE)
train_loader <- loaders$train_loader; test_loader <- loaders$test_loader
model <- SICNN_Net("regression", c(j, 5, 5, 1), input_skip = TRUE, device = "cpu")
suppressMessages(train_SICNN(epochs = 50, restarts = 1, SICNN = model, lr = 0.002,
  train_dl = train_loader, device = "cpu", n_train = i,
  epsilon_1 = 1, epsilon_T = 1e-5, steps_T = 10, sic_threshold = 0.5))
x <- train_loader$dataset$tensors[[1]]; data <- x[5, ]

cat("--- Testing block-diagonal uncertainty computation ---\n")
out_block <- tryCatch({
  get_local_explanations_gradient(model, data, num_samples=1,
    uncertainty=TRUE, fisher_dataloader=test_loader, 
    covariance_type="block-diagonal")
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  NULL
})

if (!is.null(out_block)) {
  cat("SUCCESS\n")
  cat("Block SE (first 5):", paste(round(as.numeric(out_block$se)[1:5], 6), collapse = ", "), "\n")
}

cat("--- Testing plot with block-diagonal uncertainty ---\n")
tryCatch({
  plot(model, type = "local", data = data,
       uncertainty = TRUE, fisher_dataloader = test_loader,
       covariance_type = "block-diagonal")
  cat("PLOT SUCCESS\n")
}, error = function(e) {
  cat("PLOT ERROR:", conditionMessage(e), "\n")
})
