test_that("binary SIC loss uses the -2 log-likelihood scale", {
  testthat::skip_on_cran()
  if (!requireNamespace("torch", quietly = TRUE)) {
    testthat::skip("torch not available")
  }

  torch::torch_manual_seed(1)
  model <- SICNN_Net(
    problem_type = "binary classification",
    sizes = c(1L, 1L, 1L),
    input_skip = FALSE,
    device = "cpu"
  )

  zero_layer <- function(layer) {
    layer$weight_mean$data()$zero_()
    if (layer$has_bias) {
      layer$bias_mean$data()$zero_()
    }
  }
  for (layer in model$layers$children) {
    zero_layer(layer)
  }
  zero_layer(model$out_layer)

  x <- torch::torch_zeros(c(2L, 1L))
  y <- torch::torch_tensor(c(0, 1), dtype = torch::torch_float())
  train_loader <- torch::dataloader(
    torch::tensor_dataset(x, y),
    batch_size = 2L,
    shuffle = FALSE
  )

  fit <- suppressMessages(train_SICNN(
    epochs = 1L,
    SICNN = model,
    lr = 1e-12,
    train_dl = train_loader,
    n_train = 2L,
    penalty = 1,
    epsilon_1 = 1,
    epsilon_T = 1,
    steps_T = 1L
  ))

  # At zero logits, BCE is 2 * log(2); SIC reports 2 * BCE on the -2 log-likelihood scale.
  expect_equal(fit$loss[[1L]], 4 * log(2), tolerance = 1e-6)
})