test_that("SICNN_Net S3 utilities work", {
  testthat::skip_on_cran()
  if (!requireNamespace("torch", quietly = TRUE)) {
    testthat::skip("torch not available")
  }
  
  # create mini model
  problem <- 'regression'
  sizes <- c(2, 3, 1)
  device <- 'cpu'
  model <- SICNN_Net(problem_type = problem, sizes = sizes, input_skip = TRUE, device = device)
  
  x <- torch::torch_randn(5, 2)
  b <- torch::torch_rand(2)
  y <- torch::torch_matmul(x, b)
  train_data <- torch::tensor_dataset(x, y)
  train_loader <- torch::dataloader(train_data, batch_size = 5)
  
  train_SICNN(epochs = 1, SICNN = model, lr = 0.01, train_dl = train_loader, n_train = 5)
  
  # print
  out_print <- capture.output(print(model))
  expect_true(any(grepl("SICNN Model Summary", out_print)))
  
  # summary
  out_sum <- summary(model, epsilon = 1e-5, threshold = 0.5)
  expect_true(is.data.frame(out_sum))
  
  # residuals
  res <- residuals(model)
  expect_equal(length(res), 5)
  
  # coef (local explanations)
  expl <- coef(model, dataset = x, output_neuron = 1, num_data = 2)
  expect_true(is.data.frame(expl))
  
  # predict
  preds <- predict(model, newdata = train_loader, draws = 2)
  expect_equal(dim(preds), c(2, 5, 1))
})
