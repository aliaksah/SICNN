test_that("active-path counts recompute for the requested threshold", {
  model <- SICNN_Net(
    problem_type = "binary classification",
    sizes = c(2L, 1L, 1L),
    input_skip = TRUE,
    custom_act = torch::nn_sigmoid()
  )

  model$layers$children$`0`$weight_mean$data()$copy_(
    torch::torch_tensor(matrix(c(0.05, 0), nrow = 1L), dtype = torch::torch_float())
  )
  model$out_layer$weight_mean$data()$copy_(
    torch::torch_tensor(matrix(c(0.05, 0, 0.05), nrow = 1L), dtype = torch::torch_float())
  )

  model$compute_paths_input_skip(epsilon = 0.1, threshold = 0.5, threshold_type = "phi")
  expect_equal(unname(model$sic_weight_counts(0.1, 0.5, "phi", active_paths = TRUE)[["active"]]), 0)

  # At phi > 0.1, the x1-hidden-output and x2-direct paths are active.
  counts <- model$sic_weight_counts(0.1, 0.1, "phi", active_paths = TRUE)
  expect_equal(unname(counts[["active"]]), 3)
  expect_equal(unname(counts[["total"]]), 5)
  expect_equal(model$sic_density_active_path(0.1, 0.1, "phi"), 3 / 5, tolerance = 1e-6)
})