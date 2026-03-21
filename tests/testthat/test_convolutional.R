test_that("SICNN_ConvNet builds and trains for 1 epoch", {
  testthat::skip_on_cran()
  if (!requireNamespace("torch", quietly = TRUE)) {
    testthat::skip("torch not available")
  }

  device <- 'cpu'
  torch::torch_manual_seed(42)

  conv_layer_1 <- SICNN_Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3, device = device)
  conv_layer_2 <- SICNN_Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, device = device)
  linear_layer_1 <- SICNN_Linear(in_features = 8 * 5 * 5, out_features = 32, device = device)
  linear_layer_2 <- SICNN_Linear(in_features = 32, out_features = 10, device = device)

  model <- SICNN_ConvNet(conv_layer_1, conv_layer_2, linear_layer_1, linear_layer_2, device)

  # dummy image data, pretend 3 samples of 1 channel 28x28
  x <- torch::torch_randn(3, 1, 28, 28)
  y <- torch::torch_tensor(c(1, 4, 9), dtype=torch::torch_long())
  train_data <- torch::tensor_dataset(x, y)
  train_loader <- torch::dataloader(train_data, batch_size = 3)

  res <- train_SICNN(epochs = 1, SICNN = model, lr = 0.01, train_dl = train_loader, n_train = 3)
  
  expect_true(length(res$loss) == 1)
  expect_true(is.numeric(res$density[1]))

  # test forward pass sparsity handling
  out <- model(x, sparse=TRUE)
  expect_equal(dim(out), c(3, 10))
  
  # test smooth param count
  k_smooth <- model$smooth_param_count(epsilon = 1e-5)
  expect_true(is.numeric(k_smooth$item()))

  # test sic_weight_counts
  counts <- model$sic_weight_counts(epsilon = 1e-5, threshold = 0.5)
  expect_true(counts["total"] > 0)
})
