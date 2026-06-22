#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(torch)
})

# SICNN analysis for the JAIR ISLaB/LBBNN Wisconsin Breast Cancer experiment.
#
# The paper uses the diagnostic Wisconsin Breast Cancer data with 30 inputs,
# pre-split into 512 training and 57 test observations, min-max scaled.
# Architecture: input-skip network with two hidden layers of 50 nodes, giving
# 5,580 penalized weights when p = 30.
#
# Smoke test:
#   Rscript uci_experiments/wbc_islab_sicnn.R --preset=smoke --workers=1
#
# Paper-style run:
#   Rscript uci_experiments/wbc_islab_sicnn.R --preset=paper --workers=1

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

read_islab_wbc_data <- function(data_dir) {
  required <- file.path(data_dir, c("X_train.txt", "X_test.txt", "Y_train.txt", "Y_test.txt"))
  missing <- required[!file.exists(required)]
  if (length(missing) > 0L) {
    stop(
      "Missing ISLaB WBC split files: ", paste(missing, collapse = ", "), "\n",
      "Point --data-dir at the JAIR repo's data/WBC directory."
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
    problem_type = "binary classification",
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
predict_binary_probs <- function(model, loader, sparse) {
  model$eval()
  prob <- numeric(0)
  y_true <- numeric(0)
  torch::with_no_grad({
    coro::loop(for (b in loader) {
      pred <- model(b[[1]], sparse = sparse)$squeeze()
      prob <- c(prob, as.numeric(pred$cpu()))
      y_true <- c(y_true, as.numeric(b[[2]]$cpu()))
    })
  })
  list(prob = prob, y_true = y_true)
}

binary_metrics <- function(y_true, prob, n_bins = 10L) {
  prob <- pmin(pmax(prob, 1e-8), 1 - 1e-8)
  pred_class <- as.numeric(prob >= 0.5)
  acc <- mean(pred_class == y_true)
  nll <- -mean(y_true * log(prob) + (1 - y_true) * log(1 - prob))

  confidence <- ifelse(prob >= 0.5, prob, 1 - prob)
  correct <- as.numeric(pred_class == y_true)
  breaks <- seq(0, 1, length.out = n_bins + 1L)
  bin_id <- cut(confidence, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  ece <- 0
  for (j in seq_len(n_bins)) {
    idx <- which(bin_id == j)
    if (length(idx) == 0L) next
    ece <- ece + length(idx) / length(y_true) * abs(mean(correct[idx]) - mean(confidence[idx]))
  }

  list(acc = acc, nll = nll, ece = ece, auc = roc_auc_binary(y_true, prob))
}

best_accuracy_threshold <- function(y_true, prob) {
  y_true <- as.numeric(y_true)
  candidates <- sort(unique(c(0, prob, 1)))
  if (length(candidates) > 1L) {
    candidates <- sort(unique(c(
      0,
      (candidates[-1L] + candidates[-length(candidates)]) / 2,
      1
    )))
  }

  accs <- vapply(
    candidates,
    function(thr) mean(as.numeric(prob >= thr) == y_true),
    numeric(1L)
  )
  best <- which.max(accs)
  list(threshold = candidates[[best]], acc = accs[[best]])
}

accuracy_at_threshold <- function(y_true, prob, threshold) {
  mean(as.numeric(prob >= threshold) == as.numeric(y_true))
}

roc_auc_binary <- function(y_true, score) {
  y_true <- as.numeric(y_true)
  n_pos <- sum(y_true == 1)
  n_neg <- sum(y_true == 0)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  ranks <- rank(score, ties.method = "average")
  (sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
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
  dense_pred <- predict_binary_probs(model, test_loader, sparse = FALSE)
  dense <- binary_metrics(dense_pred$y_true, dense_pred$prob, cfg$ece_bins)

  refit_sparse_model(model, train_loader, cfg)
  sparse_pred <- predict_binary_probs(model, test_loader, sparse = TRUE)
  sparse_train_pred <- predict_binary_probs(model, train_loader, sparse = TRUE)
  sparse <- binary_metrics(sparse_pred$y_true, sparse_pred$prob, cfg$ece_bins)
  sparse_threshold <- best_accuracy_threshold(
    sparse_train_pred$y_true,
    sparse_train_pred$prob
  )
  acc_sparse_train_threshold <- accuracy_at_threshold(
    sparse_pred$y_true,
    sparse_pred$prob,
    sparse_threshold$threshold
  )

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
    "  Rep %d/%d done in %.1fs | ACC full %.3f | ACC sparse %.3f | ACC sparse tuned %.3f | AUC sparse %.3f | used weights %d\n",
    rep_id, total_reps, elapsed, dense$acc, sparse$acc,
    acc_sparse_train_threshold, sparse$auc,
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
    lr = cfg$lr,
    scheduler = if (is.null(cfg$scheduler)) "none" else cfg$scheduler,
    sch_step_size = cfg$sch_step_size,
    sch_milestones = if (is.null(cfg$sch_milestones)) NA_character_ else paste(cfg$sch_milestones, collapse = ","),
    sch_gamma = cfg$sch_gamma,
    penalty = if (is.null(cfg$penalty)) log(cfg$n_train) else cfg$penalty,
    epsilon_1 = cfg$epsilon_1,
    epsilon_T = cfg$epsilon_T,
    sic_threshold = cfg$sic_threshold,
    acc_full = dense$acc,
    acc_sparse = sparse$acc,
    sparse_train_threshold = sparse_threshold$threshold,
    acc_sparse_train_threshold = acc_sparse_train_threshold,
    train_acc_sparse_train_threshold = sparse_threshold$acc,
    ece_full = dense$ece,
    ece_sparse = sparse$ece,
    nll_full = dense$nll,
    nll_sparse = sparse$nll,
    auc_full = dense$auc,
    auc_sparse = sparse$auc,
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
    acc_full = format_mmm(results$acc_full),
    acc_sparse = format_mmm(results$acc_sparse),
    acc_sparse_train_threshold = format_mmm(results$acc_sparse_train_threshold),
    sparse_train_threshold = format_mmm(results$sparse_train_threshold),
    used_weights = format_mmm(results$used_weights, digits = 1L),
    avg_depth = format_mmm(results$avg_depth, digits = 2L),
    max_depth = format_mmm(results$max_depth, digits = 1L),
    ece_full = format_mmm(results$ece_full),
    ece_sparse = format_mmm(results$ece_sparse),
    nll_full = format_mmm(results$nll_full),
    nll_sparse = format_mmm(results$nll_sparse),
    auc_full = format_mmm(results$auc_full),
    auc_sparse = format_mmm(results$auc_sparse),
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
      epochs = 200L,
      hidden_sizes = c(50L, 50L),
      lr = 0.01,
      sch_step_size = 100L,
      batch_size = 64L,
      workers = 1L
    )
  } else if (preset == "smoke") {
    cfg <- list(
      reps = 1L,
      epochs = 2000L,
      hidden_sizes = c(50L, 50L),
      lr = 0.01,
      sch_step_size = 100L,
      batch_size = 64L,
      workers = 1L
    )
  } else {
    stop("Unknown preset. Use smoke or paper.")
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
  cfg$test_batch_size <- as.integer(arg_value(args, "test-batch-size", 57L))
  cfg$epsilon_1 <- as.numeric(arg_value(args, "epsilon-1", 1))
  cfg$epsilon_T <- as.numeric(arg_value(args, "epsilon-T", 1e-5))
  cfg$steps_T <- as.integer(arg_value(args, "steps-T", min(100L, cfg$epochs)))
  cfg$sic_threshold <- as.numeric(arg_value(args, "sic-threshold", 0.5))
  cfg$sic_threshold_type <- arg_value(args, "sic-threshold-type", "phi")
  cfg$penalty <- arg_value(args, "penalty", 80 * log(512))
  if (!is.null(cfg$penalty)) cfg$penalty <- as.numeric(cfg$penalty)
  cfg$restarts <- as.integer(arg_value(args, "restarts", 1L))
  cfg$activation <- arg_value(args, "activation", "sigmoid")
  cfg$ece_bins <- as.integer(arg_value(args, "ece-bins", 10L))
  cfg$show_epochs <- parse_logical(arg_value(args, "show-epochs", NULL), FALSE)
  cfg$data_dir <- arg_value(args, "data-dir", file.path("C:", "tmp", "ISLaB-LBBNN-JAIR", "data", "WBC"))
  cfg$out <- arg_value(
    args,
    "out",
    file.path("uci_experiments", paste0("wbc_islab_sicnn_", preset, "_results.rds"))
  )

  if (!cfg$sic_threshold_type %in% c("phi", "abs")) {
    stop("sic-threshold-type must be 'phi' or 'abs'")
  }
  if (!is.null(cfg$scheduler) && !cfg$scheduler %in% c("step", "multi_step")) {
    stop("scheduler must be step, multi_step, or none")
  }
  if (!cfg$activation %in% c("sigmoid", "relu", "leaky_relu")) {
    stop("activation must be one of: sigmoid, relu, leaky_relu")
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

  data_obj <- read_islab_wbc_data(cfg$data_dir)
  cfg$n_train <- nrow(data_obj$x_train)
  cfg$n_test <- nrow(data_obj$x_test)
  p <- ncol(data_obj$x_train)
  cfg$workers <- max(1L, min(cfg$workers, cfg$reps))

  cat("ISLaB/LBBNN WBC split with SICNN\n")
  cat(sprintf("Preset: %s | reps: %d | data dir: %s\n", cfg$preset, cfg$reps, cfg$data_dir))
  cat(sprintf("n_train=%d | n_test=%d | p=%d | architecture=%s\n",
              cfg$n_train, cfg$n_test, p, paste(c(p, cfg$hidden_sizes, 1L), collapse = "-")))
  cat(sprintf("lr=%.4g | penalty=%s | epsilon %.2g -> %.2g | threshold=%s %.3f\n",
              cfg$lr,
              if (is.null(cfg$penalty)) "log(n_train)" else as.character(cfg$penalty),
              cfg$epsilon_1, cfg$epsilon_T, cfg$sic_threshold_type, cfg$sic_threshold))
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

  cat("\nWBC SICNN analysis complete.\n")
  cat(sprintf("Elapsed wall time: %.1f seconds (%.2f minutes)\n", elapsed_all, elapsed_all / 60))
  cat(sprintf("Final results saved to: %s\n\n", cfg$out))
  print(summary_table)
  cat("\nNonzero feature inclusion rates:\n")
  print(inclusion_table[inclusion_table$inclusion_rate > 0, , drop = FALSE])
}

main()
