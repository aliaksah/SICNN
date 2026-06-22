#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(torch)
})

# SICNN analysis for the JAIR ISLaB/LBBNN Abalone experiment.
#
# The paper uses the Abalone regression data with 9 inputs, pre-split into
# 3,759 training and 418 test observations. Inputs are already scaled in the
# JAIR repo split, and the target is the raw number of rings.
# Architecture: input-skip network with two hidden layers of 200 nodes, giving
# 43,809 penalized weights when p = 9.
#
# Smoke test:
#   Rscript uci_experiments/abalone_islab_sicnn.R --preset=smoke --workers=1
#
# Paper-style run:
#   Rscript uci_experiments/abalone_islab_sicnn.R --preset=paper --workers=1

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    stripped <- sub("^--", "", arg)
    if (grepl("=", stripped, fixed = TRUE)) {
      parts <- strsplit(stripped, "=", fixed = TRUE)[[1]]
      out[[parts[[1]]]] <- paste(parts[-1], collapse = "=")
    } else {
      out[[stripped]] <- TRUE
    }
  }
  out
}

arg_value <- function(args, name, default = NULL) {
  if (!is.null(args[[name]])) args[[name]] else default
}

parse_int_vec <- function(x, default) {
  if (is.null(x)) return(default)
  as.integer(strsplit(x, ",", fixed = TRUE)[[1]])
}

parse_logical <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y")
}

load_sicnn <- function() {
  if (requireNamespace("devtools", quietly = TRUE) && file.exists("DESCRIPTION")) {
    suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
  } else {
    suppressPackageStartupMessages(library(SICNN))
  }
}

read_vector <- function(path) {
  as.numeric(read.table(path, sep = ",", header = FALSE)[[1]])
}

read_islab_abalone_data <- function(data_dir) {
  required <- file.path(data_dir, c("X_train.txt", "X_test.txt", "Y_train.txt", "Y_test.txt"))
  missing <- required[!file.exists(required)]
  if (length(missing) > 0L) {
    stop(
      "Missing ISLaB Abalone split files: ", paste(missing, collapse = ", "), "\n",
      "Point --data-dir at the JAIR repo's data/abalone directory."
    )
  }

  x_train <- as.matrix(read.table(required[[1L]], sep = ",", header = FALSE))
  x_test <- as.matrix(read.table(required[[2L]], sep = ",", header = FALSE))
  y_train <- read_vector(required[[3L]])
  y_test <- read_vector(required[[4L]])

  colnames(x_train) <- paste0("x", seq_len(ncol(x_train)))
  colnames(x_test) <- colnames(x_train)

  list(
    x_train = x_train,
    x_test = x_test,
    y_train = y_train,
    y_test = y_test
  )
}

make_loader <- function(x, y, batch_size, shuffle) {
  ds <- torch::tensor_dataset(
    torch::torch_tensor(x, dtype = torch::torch_float()),
    torch::torch_tensor(as.numeric(y), dtype = torch::torch_float())
  )
  torch::dataloader(ds, batch_size = min(batch_size, nrow(x)), shuffle = shuffle)
}

fit_sicnn <- function(train_loader, cfg, p, seed) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  activation <- switch(
    cfg$activation,
    sigmoid = torch::nn_sigmoid(),
    relu = torch::nn_relu(),
    leaky_relu = torch::nn_leaky_relu(0.00),
    stop("Unknown activation: ", cfg$activation)
  )

  model <- SICNN_Net(
    problem_type = "regression",
    sizes = c(p, cfg$hidden_sizes, 1L),
    input_skip = TRUE,
    device = "cpu",
    custom_act = activation
  )

  train_call <- function() {
    train_SICNN(
      epochs = cfg$epochs,
      restarts = cfg$restarts,
      SICNN = model,
      lr = cfg$lr,
      train_dl = train_loader,
      device = "cpu",
      scheduler = cfg$scheduler,
      sch_step_size = cfg$sch_step_size,
      sch_milestones = cfg$sch_milestones,
      sch_gamma = cfg$sch_gamma,
      n_train = cfg$n_train,
      epsilon_1 = cfg$epsilon_1,
      epsilon_T = cfg$epsilon_T,
      steps_T = cfg$steps_T,
      sic_threshold = cfg$sic_threshold,
      sic_threshold_type = cfg$sic_threshold_type,
      penalty = cfg$penalty
    )
  }

  if (isTRUE(cfg$show_epochs)) {
    train_call()
  } else {
    suppressMessages(train_call())
  }

  model
}

refit_sparse_model <- function(model, train_loader, cfg) {
  if (cfg$post_refit_epochs <= 0L) return(invisible(model))

  optimizer <- torch::optim_adam(model$parameters, lr = cfg$post_refit_lr)
  for (epoch in seq_len(cfg$post_refit_epochs)) {
    model$train()
    coro::loop(for (b in train_loader) {
      optimizer$zero_grad()
      output <- model(b[[1]], sparse = TRUE)$squeeze()
      target <- b[[2]]
      loss <- model$loss_fn(output, target) / dim(b[[1]])[1L]
      loss$backward()
      optimizer$step()
    })
  }
  invisible(model)
}

predict_numeric <- function(model, loader, sparse) {
  model$eval()
  pred <- numeric(0)
  y_true <- numeric(0)
  torch::with_no_grad({
    coro::loop(for (b in loader) {
      out <- model(b[[1]], sparse = sparse)$squeeze()
      pred <- c(pred, as.numeric(out$cpu()))
      y_true <- c(y_true, as.numeric(b[[2]]$cpu()))
    })
  })
  list(pred = pred, y_true = y_true)
}

pinball_loss <- function(y_true, pred, alpha = 0.5) {
  err <- y_true - pred
  mean(pmax(alpha * err, (alpha - 1) * err))
}

affine_calibration <- function(y_true, pred) {
  fit <- stats::lm(y_true ~ pred)
  coefs <- stats::coef(fit)
  list(intercept = unname(coefs[[1L]]), slope = unname(coefs[[2L]]))
}

apply_affine_calibration <- function(pred, calibration) {
  calibration$intercept + calibration$slope * pred
}

regression_metrics <- function(y_true, pred) {
  rmse <- sqrt(mean((pred - y_true)^2))
  tss <- sum((y_true - mean(y_true))^2)
  r2 <- 1 - sum((pred - y_true)^2) / tss
  corr <- if (stats::sd(pred) == 0 || stats::sd(y_true) == 0) NA_real_ else stats::cor(y_true, pred)
  list(
    rmse = rmse,
    r2 = r2,
    corr = corr,
    pinball = pinball_loss(y_true, pred, alpha = 0.5)
  )
}

select_sicnn_features <- function(model, p, epsilon_T, sic_threshold, sic_threshold_type) {
  model$compute_paths_input_skip(
    epsilon = epsilon_T,
    threshold = sic_threshold,
    threshold_type = sic_threshold_type
  )
  selected <- rep(FALSE, p)

  for (layer in model$layers$children) {
    alpha <- as.matrix(layer$alpha_active_path$cpu())
    in_features <- ncol(alpha)
    cov_cols <- if (in_features == p) seq_len(p) else (in_features - p + 1L):in_features
    selected <- selected | colSums(alpha[, cov_cols, drop = FALSE]) > 0
  }

  alpha_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  in_features <- ncol(alpha_out)
  cov_cols <- if (in_features == p) seq_len(p) else (in_features - p + 1L):in_features
  selected <- selected | colSums(alpha_out[, cov_cols, drop = FALSE]) > 0

  selected
}

path_depth_metrics <- function(model, epsilon_T, sic_threshold, sic_threshold_type) {
  model$compute_paths_input_skip(
    epsilon = epsilon_T,
    threshold = sic_threshold,
    threshold_type = sic_threshold_type
  )

  node_depths <- NULL
  for (layer in model$layers$children) {
    alpha <- as.matrix(layer$alpha_active_path$cpu())
    layer_depths <- vector("list", nrow(alpha))

    for (out_node in seq_len(nrow(alpha))) {
      depths <- integer(0)
      active_cols <- which(alpha[out_node, ] > 0)
      for (col in active_cols) {
        if (is.null(node_depths) || col > length(node_depths)) {
          depths <- c(depths, 1L)
        } else if (length(node_depths[[col]]) > 0L) {
          depths <- c(depths, node_depths[[col]] + 1L)
        }
      }
      layer_depths[[out_node]] <- depths
    }
    node_depths <- layer_depths
  }

  alpha_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  output_depths <- integer(0)
  for (out_node in seq_len(nrow(alpha_out))) {
    active_cols <- which(alpha_out[out_node, ] > 0)
    for (col in active_cols) {
      if (is.null(node_depths) || col > length(node_depths)) {
        output_depths <- c(output_depths, 1L)
      } else if (length(node_depths[[col]]) > 0L) {
        output_depths <- c(output_depths, node_depths[[col]] + 1L)
      }
    }
  }

  if (length(output_depths) == 0L) {
    return(list(avg_depth = NA_real_, max_depth = NA_real_, n_active_paths = 0L))
  }

  list(
    avg_depth = mean(output_depths),
    max_depth = max(output_depths),
    n_active_paths = length(output_depths)
  )
}

run_one_rep <- function(rep_id, cfg, data_obj, total_reps) {
  load_sicnn()
  torch::torch_set_num_threads(cfg$torch_threads)

  p <- ncol(data_obj$x_train)
  seed <- cfg$seed + rep_id - 1L

  cat(sprintf(
    "Rep %d/%d | seed=%d | n_train=%d | n_test=%d | epochs=%d\n",
    rep_id, total_reps, seed, cfg$n_train, cfg$n_test, cfg$epochs
  ))
  flush.console()

  train_loader <- make_loader(data_obj$x_train, data_obj$y_train, cfg$batch_size, shuffle = TRUE)
  test_loader <- make_loader(data_obj$x_test, data_obj$y_test, cfg$test_batch_size, shuffle = FALSE)

  started <- proc.time()[[3]]
  model <- fit_sicnn(train_loader, cfg = cfg, p = p, seed = seed)

  model$compute_paths_input_skip(
    epsilon = cfg$epsilon_T,
    threshold = cfg$sic_threshold,
    threshold_type = cfg$sic_threshold_type
  )

  dense_pred <- predict_numeric(model, test_loader, sparse = FALSE)
  dense <- regression_metrics(dense_pred$y_true, dense_pred$pred)

  refit_sparse_model(model, train_loader, cfg)

  sparse_pred <- predict_numeric(model, test_loader, sparse = TRUE)
  sparse_train_pred <- predict_numeric(model, train_loader, sparse = TRUE)
  sparse_calibration <- affine_calibration(
    sparse_train_pred$y_true,
    sparse_train_pred$pred
  )
  sparse_calibrated_pred <- apply_affine_calibration(
    sparse_pred$pred,
    sparse_calibration
  )
  sparse <- regression_metrics(sparse_pred$y_true, sparse_pred$pred)
  sparse_calibrated <- regression_metrics(sparse_pred$y_true, sparse_calibrated_pred)

  sic_counts <- model$sic_weight_counts(
    epsilon = cfg$epsilon_T,
    threshold = cfg$sic_threshold,
    threshold_type = cfg$sic_threshold_type,
    active_paths = TRUE
  )
  selected <- select_sicnn_features(
    model,
    p = p,
    epsilon_T = cfg$epsilon_T,
    sic_threshold = cfg$sic_threshold,
    sic_threshold_type = cfg$sic_threshold_type
  )
  depth <- path_depth_metrics(
    model,
    epsilon_T = cfg$epsilon_T,
    sic_threshold = cfg$sic_threshold,
    sic_threshold_type = cfg$sic_threshold_type
  )

  elapsed <- proc.time()[[3]] - started
  cat(sprintf(
    "  Rep %d/%d done in %.1fs | RMSE full %.3f | RMSE sparse %.3f | RMSE sparse cal %.3f | Corr sparse %.3f | used weights %d\n",
    rep_id, total_reps, elapsed, dense$rmse, sparse$rmse,
    sparse_calibrated$rmse, sparse$corr,
    as.integer(sic_counts[["active"]])
  ))
  flush.console()

  data.frame(
    method = "sicnn_smooth_l0",
    rep = rep_id,
    seed = seed,
    n_train = cfg$n_train,
    n_test = cfg$n_test,
    epochs = cfg$epochs,
    post_refit_epochs = cfg$post_refit_epochs,
    post_refit_lr = cfg$post_refit_lr,
    hidden_sizes = paste(cfg$hidden_sizes, collapse = "-"),
    activation = cfg$activation,
    lr = cfg$lr,
    scheduler = if (is.null(cfg$scheduler)) "none" else cfg$scheduler,
    sch_step_size = cfg$sch_step_size,
    sch_milestones = if (is.null(cfg$sch_milestones)) NA_character_ else paste(cfg$sch_milestones, collapse = ","),
    sch_gamma = cfg$sch_gamma,
    penalty = if (is.null(cfg$penalty)) log(cfg$n_train) else cfg$penalty,
    epsilon_1 = cfg$epsilon_1,
    epsilon_T = cfg$epsilon_T,
    sic_threshold = cfg$sic_threshold,
    sic_threshold_type = cfg$sic_threshold_type,
    rmse_full = dense$rmse,
    rmse_sparse = sparse$rmse,
    rmse_sparse_affine = sparse_calibrated$rmse,
    corr_full = dense$corr,
    corr_sparse = sparse$corr,
    corr_sparse_affine = sparse_calibrated$corr,
    r2_full = dense$r2,
    r2_sparse = sparse$r2,
    r2_sparse_affine = sparse_calibrated$r2,
    pinball_full = dense$pinball,
    pinball_sparse = sparse$pinball,
    pinball_sparse_affine = sparse_calibrated$pinball,
    sparse_affine_intercept = sparse_calibration$intercept,
    sparse_affine_slope = sparse_calibration$slope,
    used_weights = as.numeric(sic_counts[["active"]]),
    total_weights = as.numeric(sic_counts[["total"]]),
    sparsity_pct = as.numeric(sic_counts[["removed"]] / sic_counts[["total"]]) * 100,
    avg_depth = depth$avg_depth,
    max_depth = depth$max_depth,
    n_active_paths = depth$n_active_paths,
    selected_count = sum(selected),
    selected_features = paste(which(selected), collapse = ","),
    elapsed_seconds = elapsed,
    stringsAsFactors = FALSE
  )
}

format_mmm <- function(x, digits = 3L) {
  if (all(is.na(x))) return(NA_character_)
  fmt <- paste0("%.", digits, "f (%.", digits, "f, %.", digits, "f)")
  sprintf(fmt, stats::median(x, na.rm = TRUE), min(x, na.rm = TRUE), max(x, na.rm = TRUE))
}

summarise_results <- function(results) {
  data.frame(
    method = "sicnn_smooth_l0",
    reps = nrow(results),
    corr_full = format_mmm(results$corr_full),
    corr_sparse = format_mmm(results$corr_sparse),
    rmse_full = format_mmm(results$rmse_full),
    rmse_sparse = format_mmm(results$rmse_sparse),
    rmse_sparse_affine = format_mmm(results$rmse_sparse_affine),
    pinball_full = format_mmm(results$pinball_full),
    pinball_sparse = format_mmm(results$pinball_sparse),
    pinball_sparse_affine = format_mmm(results$pinball_sparse_affine),
    used_weights = format_mmm(results$used_weights, digits = 1L),
    avg_depth = format_mmm(results$avg_depth, digits = 2L),
    max_depth = format_mmm(results$max_depth, digits = 1L),
    mean_elapsed_seconds = mean(results$elapsed_seconds),
    stringsAsFactors = FALSE
  )
}

summarise_inclusion_rates <- function(results, p) {
  rates <- numeric(p)
  for (j in seq_len(p)) {
    rates[[j]] <- mean(vapply(
      strsplit(results$selected_features, ",", fixed = TRUE),
      function(x) as.character(j) %in% x,
      logical(1L)
    ))
  }
  data.frame(feature = paste0("x", seq_len(p)), inclusion_rate = rates)
}

make_config <- function(args) {
  preset <- arg_value(args, "preset", "smoke")

  if (preset == "paper") {
    cfg <- list(
      reps = 10L,
      epochs = 5000L,
      hidden_sizes = c(200L, 200L),
      lr = 0.01,
      sch_step_size = 1250L,
      batch_size = 751L,
      workers = 1L
    )
  } else if (preset == "tuning") {
    cfg <- list(
      reps = 1L,
      epochs = 500L,
      hidden_sizes = c(200L, 200L),
      lr = 0.005,
      sch_step_size = 250L,
      batch_size = 751L,
      workers = 1L
    )
  } else if (preset == "smoke") {
    cfg <- list(
      reps = 1L,
      epochs = 2L,
      hidden_sizes = c(5L, 5L),
      lr = 0.01,
      sch_step_size = 1L,
      batch_size = 128L,
      workers = 1L
    )
  } else {
    stop("Unknown preset. Use smoke, tuning, or paper.")
  }

  cfg$preset <- preset
  cfg$reps <- as.integer(arg_value(args, "reps", cfg$reps))
  cfg$epochs <- as.integer(arg_value(args, "epochs", cfg$epochs))
  cfg$post_refit_epochs <- as.integer(arg_value(args, "post-refit-epochs", 0L))
  cfg$post_refit_lr <- as.numeric(arg_value(args, "post-refit-lr", 0.01))
  cfg$hidden_sizes <- parse_int_vec(arg_value(args, "hidden-sizes"), cfg$hidden_sizes)
  cfg$lr <- as.numeric(arg_value(args, "lr", cfg$lr))
  cfg$scheduler <- arg_value(args, "scheduler", "step")
  if (cfg$scheduler == "none") cfg$scheduler <- NULL
  cfg$sch_step_size <- as.integer(arg_value(args, "sch-step-size", cfg$sch_step_size))
  cfg$sch_milestones <- arg_value(args, "sch-milestones", NULL)
  if (!is.null(cfg$sch_milestones)) cfg$sch_milestones <- as.numeric(strsplit(cfg$sch_milestones, ",", fixed = TRUE)[[1]])
  cfg$sch_gamma <- as.numeric(arg_value(args, "sch-gamma", if (is.null(cfg$scheduler)) 0.1 else if (cfg$scheduler == "multi_step") 0.5 else 0.1))
  cfg$batch_size <- as.integer(arg_value(args, "batch-size", cfg$batch_size))
  cfg$workers <- as.integer(arg_value(args, "workers", cfg$workers))
  cfg$seed <- as.integer(arg_value(args, "seed", 42L))
  cfg$torch_threads <- as.integer(arg_value(args, "torch-threads", 1L))
  cfg$test_batch_size <- as.integer(arg_value(args, "test-batch-size", 418L))
  cfg$epsilon_1 <- as.numeric(arg_value(args, "epsilon-1", 0.05))
  cfg$epsilon_T <- as.numeric(arg_value(args, "epsilon-T", 0.005))
  cfg$steps_T <- as.integer(arg_value(args, "steps-T", min(100L, cfg$epochs)))
  cfg$sic_threshold <- as.numeric(arg_value(args, "sic-threshold", 0.5))
  cfg$sic_threshold_type <- arg_value(args, "sic-threshold-type", "phi")
  cfg$penalty <- arg_value(args, "penalty", NULL)
  if (!is.null(cfg$penalty)) cfg$penalty <- as.numeric(cfg$penalty)
  cfg$restarts <- as.integer(arg_value(args, "restarts", 1L))
  cfg$activation <- arg_value(args, "activation", "sigmoid")
  cfg$show_epochs <- parse_logical(arg_value(args, "show-epochs", NULL), FALSE)
  cfg$data_dir <- arg_value(args, "data-dir", file.path("C:", "tmp", "ISLaB-LBBNN-JAIR", "data", "abalone"))
  cfg$out <- arg_value(
    args,
    "out",
    file.path("uci_experiments", paste0("abalone_islab_sicnn_", preset, "_results.rds"))
  )

  if (!cfg$sic_threshold_type %in% c("phi", "abs")) {
    stop("sic-threshold-type must be 'phi' or 'abs'")
  }
  if (!cfg$activation %in% c("sigmoid", "relu", "leaky_relu")) {
    stop("activation must be one of: sigmoid, relu, leaky_relu")
  }
  if (!is.null(cfg$scheduler) && !cfg$scheduler %in% c("step", "multi_step")) {
    stop("scheduler must be step, multi_step, or none")
  }
  if (length(cfg$hidden_sizes) < 1L) {
    stop("At least one hidden layer is required")
  }
  cfg
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  cfg <- make_config(args)

  load_sicnn()
  torch::torch_set_num_threads(cfg$torch_threads)

  data_obj <- read_islab_abalone_data(cfg$data_dir)
  cfg$n_train <- nrow(data_obj$x_train)
  cfg$n_test <- nrow(data_obj$x_test)
  p <- ncol(data_obj$x_train)
  cfg$workers <- max(1L, min(cfg$workers, cfg$reps))

  cat("ISLaB/LBBNN Abalone split with SICNN\n")
  cat(sprintf("Preset: %s | reps: %d | data dir: %s\n", cfg$preset, cfg$reps, cfg$data_dir))
  cat(sprintf("n_train=%d | n_test=%d | p=%d | architecture=%s\n",
              cfg$n_train, cfg$n_test, p, paste(c(p, cfg$hidden_sizes, 1L), collapse = "-")))
  cat(sprintf("activation=%s | lr=%.4g | scheduler=%s | step=%d | gamma=%.3g | penalty=%s | epsilon %.2g -> %.2g | threshold=%s %.3f | refit=%d at lr %.4g\n",
              cfg$activation,
              cfg$lr,
              if (is.null(cfg$scheduler)) "none" else cfg$scheduler,
              cfg$sch_step_size,
              cfg$sch_gamma,
              if (is.null(cfg$penalty)) "log(n_train)" else as.character(cfg$penalty),
              cfg$epsilon_1, cfg$epsilon_T, cfg$sic_threshold_type, cfg$sic_threshold,
              cfg$post_refit_epochs, cfg$post_refit_lr))
  cat(sprintf("Output path: %s\n\n", cfg$out))

  start_all <- proc.time()[[3]]
  rep_ids <- seq_len(cfg$reps)
  if (cfg$workers == 1L) {
    chunks <- vector("list", cfg$reps)
    for (i in rep_ids) {
      chunks[[i]] <- run_one_rep(i, cfg = cfg, data_obj = data_obj, total_reps = cfg$reps)
      partial <- do.call(rbind, chunks[seq_len(i)])
      attr(partial, "config") <- cfg
      dir.create(dirname(cfg$out), recursive = TRUE, showWarnings = FALSE)
      saveRDS(partial, cfg$out)
      cat(sprintf("Progress: %d/%d reps complete. Partial saved.\n", i, cfg$reps))
      flush.console()
    }
  } else {
    cluster <- parallel::makeCluster(cfg$workers, outfile = "")
    on.exit(parallel::stopCluster(cluster), add = TRUE)
    parallel::clusterExport(
      cluster,
      varlist = setdiff(ls(envir = .GlobalEnv), "args"),
      envir = .GlobalEnv
    )
    parallel::clusterCall(cluster, setwd, getwd())
    chunks <- parallel::parLapplyLB(
      cluster,
      rep_ids,
      run_one_rep,
      cfg = cfg,
      data_obj = data_obj,
      total_reps = cfg$reps
    )
  }

  results <- do.call(rbind, chunks)
  elapsed_all <- proc.time()[[3]] - start_all
  attr(results, "config") <- cfg
  attr(results, "elapsed_seconds") <- elapsed_all

  summary_table <- summarise_results(results)
  inclusion_table <- summarise_inclusion_rates(results, p)

  dir.create(dirname(cfg$out), recursive = TRUE, showWarnings = FALSE)
  saveRDS(results, cfg$out)
  write.csv(summary_table, sub("[.]rds$", "_summary.csv", cfg$out), row.names = FALSE)
  write.csv(inclusion_table, sub("[.]rds$", "_inclusion_rates.csv", cfg$out), row.names = FALSE)

  cat("\nAbalone SICNN analysis complete.\n")
  cat(sprintf("Elapsed wall time: %.1f seconds (%.2f minutes)\n", elapsed_all, elapsed_all / 60))
  cat(sprintf("Final results saved to: %s\n\n", cfg$out))
  print(summary_table)
  cat("\nNonzero feature inclusion rates:\n")
  print(inclusion_table[inclusion_table$inclusion_rate > 0, , drop = FALSE])
}

main()