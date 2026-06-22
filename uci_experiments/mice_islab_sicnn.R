#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(torch))

# SICNN replication scaffold for the JAIR ISLaB/LBBNN Mice Protein Expression data.
# The supplied split has 972 training and 108 test observations, 77 inputs, and
# eight classes. LBBNN uses four sigmoid hidden layers of width 100, 2,000
# epochs, a batch size of floor(972 / 27) = 36, and a multi-step schedule.

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    item <- sub("^--", "", arg)
    if (grepl("=", item, fixed = TRUE)) {
      parts <- strsplit(item, "=", fixed = TRUE)[[1L]]
      out[[parts[[1L]]]] <- paste(parts[-1L], collapse = "=")
    } else {
      out[[item]] <- TRUE
    }
  }
  out
}

arg_value <- function(args, name, default = NULL) {
  if (!is.null(args[[name]])) args[[name]] else default
}

parse_logical <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y")
}

parse_int_vec <- function(x, default) {
  if (is.null(x)) return(default)
  as.integer(strsplit(x, ",", fixed = TRUE)[[1L]])
}

load_sicnn <- function() {
  if (requireNamespace("devtools", quietly = TRUE) && file.exists("DESCRIPTION")) {
    suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
  } else {
    suppressPackageStartupMessages(library(SICNN))
  }
}

read_vector <- function(path) {
  as.numeric(read.table(path, sep = ",", header = FALSE)[[1L]])
}

read_islab_mice_data <- function(data_dir, standardize) {
  required <- file.path(data_dir, c("X_train.txt", "X_test.txt", "Y_train.txt", "Y_test.txt"))
  missing <- required[!file.exists(required)]
  if (length(missing) > 0L) {
    stop("Missing ISLaB Mice split files: ", paste(missing, collapse = ", "))
  }

  x_train <- as.matrix(read.table(required[[1L]], sep = ",", header = FALSE))
  x_test <- as.matrix(read.table(required[[2L]], sep = ",", header = FALSE))
  y_train <- read_vector(required[[3L]])
  y_test <- read_vector(required[[4L]])

  classes <- sort(unique(y_train))
  if (!identical(classes, sort(unique(y_test)))) stop("Train/test class labels differ")
  if (!isTRUE(all(classes == seq(0, length(classes) - 1)))) {
    stop("Expected zero-indexed consecutive class labels in the supplied split")
  }

  if (standardize) {
    center <- colMeans(x_train)
    scale <- apply(x_train, 2L, stats::sd)
    if (any(!is.finite(scale) | scale == 0)) stop("Cannot standardize zero-variance features")
    x_train <- sweep(sweep(x_train, 2L, center, "-"), 2L, scale, "/")
    x_test <- sweep(sweep(x_test, 2L, center, "-"), 2L, scale, "/")
  }

  list(
    x_train = x_train,
    x_test = x_test,
    # torch NLL uses one-indexed class targets in this package.
    y_train = as.integer(y_train) + 1L,
    y_test = as.integer(y_test) + 1L,
    n_classes = length(classes)
  )
}

make_loader <- function(x, y, batch_size, shuffle) {
  dataset <- torch::tensor_dataset(
    torch::torch_tensor(x, dtype = torch::torch_float()),
    torch::torch_tensor(y, dtype = torch::torch_long())
  )
  torch::dataloader(dataset, batch_size = min(batch_size, nrow(x)), shuffle = shuffle)
}

fit_sicnn <- function(train_loader, cfg, p, n_classes, seed) {
  set.seed(seed)
  torch::torch_manual_seed(seed)

  model <- SICNN_Net(
    problem_type = "multiclass classification",
    sizes = c(p, cfg$hidden_sizes, n_classes),
    input_skip = TRUE,
    device = "cpu",
    custom_act = torch::nn_sigmoid()
  )

  train_call <- function() {
    train_SICNN(
      epochs = cfg$epochs,
      restarts = 1L,
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

  if (cfg$show_epochs) train_call() else suppressMessages(train_call())
  model
}

refit_sparse_model <- function(model, train_loader, cfg) {
  if (cfg$post_refit_epochs <= 0L) return(invisible(model))

  optimizer <- torch::optim_adam(model$parameters, lr = cfg$post_refit_lr)
  for (epoch in seq_len(cfg$post_refit_epochs)) {
    model$train()
    coro::loop(for (batch in train_loader) {
      optimizer$zero_grad()
      output <- model(batch[[1]], sparse = TRUE)
      target <- batch[[2]]$to(dtype = torch::torch_long())
      loss <- model$loss_fn(output, target) / dim(batch[[1]])[1L]
      loss$backward()
      optimizer$step()
    })
  }
  invisible(model)
}

predict_multiclass <- function(model, loader, sparse) {
  model$eval()
  prediction <- integer(0)
  y_true <- integer(0)
  torch::with_no_grad({
    coro::loop(for (batch in loader) {
      log_prob <- model(batch[[1]], sparse = sparse)
      prediction <- c(prediction, max.col(as.matrix(log_prob$cpu())))
      y_true <- c(y_true, as.integer(batch[[2]]$cpu()))
    })
  })
  list(prediction = prediction, y_true = y_true)
}

classification_metrics <- function(y_true, prediction) {
  accuracy <- mean(y_true == prediction)
  classes <- sort(unique(c(y_true, prediction)))
  per_class <- vapply(classes, function(k) {
    idx <- y_true == k
    if (!any(idx)) return(NA_real_)
    mean(prediction[idx] == k)
  }, numeric(1L))
  list(accuracy = accuracy, balanced_accuracy = mean(per_class, na.rm = TRUE))
}

selected_features <- function(model, p) {
  selected <- rep(FALSE, p)
  for (layer in model$layers$children) {
    alpha <- as.matrix(layer$alpha_active_path$cpu())
    input_columns <- if (ncol(alpha) == p) seq_len(p) else (ncol(alpha) - p + 1L):ncol(alpha)
    selected <- selected | colSums(alpha[, input_columns, drop = FALSE]) > 0
  }
  alpha_out <- as.matrix(model$out_layer$alpha_active_path$cpu())
  input_columns <- if (ncol(alpha_out) == p) seq_len(p) else (ncol(alpha_out) - p + 1L):ncol(alpha_out)
  selected | colSums(alpha_out[, input_columns, drop = FALSE]) > 0
}

run_one_rep <- function(rep_id, cfg, data_obj, total_reps) {
  load_sicnn()
  torch::torch_set_num_threads(cfg$torch_threads)
  seed <- cfg$seed + rep_id - 1L
  p <- ncol(data_obj$x_train)

  cat(sprintf("Rep %d/%d | seed=%d | epochs=%d\n", rep_id, total_reps, seed, cfg$epochs))
  flush.console()
  train_loader <- make_loader(data_obj$x_train, data_obj$y_train, cfg$batch_size, shuffle = TRUE)
  test_loader <- make_loader(data_obj$x_test, data_obj$y_test, cfg$test_batch_size, shuffle = FALSE)

  started <- proc.time()[[3L]]
  model <- fit_sicnn(train_loader, cfg, p, data_obj$n_classes, seed)
  model$compute_paths_input_skip(
    epsilon = cfg$epsilon_T,
    threshold = cfg$sic_threshold,
    threshold_type = cfg$sic_threshold_type
  )

  # Record the SIC-selected graph before refitting. The mask is then fixed.
  selected_count <- model$sic_weight_counts(
    epsilon = cfg$epsilon_T,
    threshold = cfg$sic_threshold,
    threshold_type = cfg$sic_threshold_type,
    active_paths = TRUE
  )
  selected_inputs <- selected_features(model, p)
  dense_prediction <- predict_multiclass(model, test_loader, sparse = FALSE)
  dense_metrics <- classification_metrics(dense_prediction$y_true, dense_prediction$prediction)

  refit_sparse_model(model, train_loader, cfg)
  sparse_prediction <- predict_multiclass(model, test_loader, sparse = TRUE)
  sparse_metrics <- classification_metrics(sparse_prediction$y_true, sparse_prediction$prediction)

  elapsed <- proc.time()[[3L]] - started
  cat(sprintf(
    "  Rep %d/%d done in %.1fs | ACC dense %.3f | ACC sparse-refit %.3f | used weights %d\n",
    rep_id, total_reps, elapsed, dense_metrics$accuracy, sparse_metrics$accuracy,
    as.integer(selected_count[["active"]])
  ))
  flush.console()

  data.frame(
    method = "sicnn_smooth_l0",
    rep = rep_id,
    seed = seed,
    n_train = cfg$n_train,
    n_test = cfg$n_test,
    n_classes = data_obj$n_classes,
    standardized_inputs = cfg$standardize,
    epochs = cfg$epochs,
    hidden_sizes = paste(cfg$hidden_sizes, collapse = "-"),
    lr = cfg$lr,
    scheduler = cfg$scheduler,
    sch_milestones = paste(cfg$sch_milestones, collapse = ","),
    sch_gamma = cfg$sch_gamma,
    penalty = cfg$penalty,
    epsilon_1 = cfg$epsilon_1,
    epsilon_T = cfg$epsilon_T,
    steps_T = cfg$steps_T,
    sic_threshold = cfg$sic_threshold,
    sic_threshold_type = cfg$sic_threshold_type,
    post_refit_epochs = cfg$post_refit_epochs,
    post_refit_lr = cfg$post_refit_lr,
    acc_dense = dense_metrics$accuracy,
    balanced_acc_dense = dense_metrics$balanced_accuracy,
    acc_sparse_refit = sparse_metrics$accuracy,
    balanced_acc_sparse_refit = sparse_metrics$balanced_accuracy,
    used_weights = as.numeric(selected_count[["active"]]),
    total_weights = as.numeric(selected_count[["total"]]),
    selected_count = sum(selected_inputs),
    selected_features = paste(which(selected_inputs), collapse = ","),
    elapsed_seconds = elapsed,
    stringsAsFactors = FALSE
  )
}

format_mmm <- function(x, digits = 3L) {
  sprintf(paste0("%.", digits, "f (%.", digits, "f, %.", digits, "f)"),
          stats::median(x), min(x), max(x))
}

summarise_results <- function(results) {
  data.frame(
    method = "sicnn_smooth_l0",
    reps = nrow(results),
    acc_dense = format_mmm(results$acc_dense),
    acc_sparse_refit = format_mmm(results$acc_sparse_refit),
    balanced_acc_sparse_refit = format_mmm(results$balanced_acc_sparse_refit),
    used_weights = format_mmm(results$used_weights, digits = 1L),
    selected_inputs = format_mmm(results$selected_count, digits = 1L),
    mean_elapsed_seconds = mean(results$elapsed_seconds),
    stringsAsFactors = FALSE
  )
}

make_config <- function(args) {
  preset <- arg_value(args, "preset", "smoke")
  if (preset == "smoke") {
    cfg <- list(reps = 1L, epochs = 2L, hidden_sizes = c(20L, 20L), lr = 0.01, batch_size = 36L)
  } else if (preset == "pilot") {
    cfg <- list(reps = 1L, epochs = 2000L, hidden_sizes = rep(100L, 4L), lr = 0.01, batch_size = 36L)
  } else if (preset == "paper") {
    cfg <- list(reps = 10L, epochs = 2000L, hidden_sizes = rep(100L, 4L), lr = 0.01, batch_size = 36L)
  } else {
    stop("preset must be smoke, pilot, or paper")
  }

  cfg$preset <- preset
  cfg$reps <- as.integer(arg_value(args, "reps", cfg$reps))
  cfg$epochs <- as.integer(arg_value(args, "epochs", cfg$epochs))
  cfg$hidden_sizes <- parse_int_vec(arg_value(args, "hidden-sizes"), cfg$hidden_sizes)
  cfg$lr <- as.numeric(arg_value(args, "lr", cfg$lr))
  cfg$batch_size <- as.integer(arg_value(args, "batch-size", cfg$batch_size))
  cfg$test_batch_size <- as.integer(arg_value(args, "test-batch-size", 108L))
  cfg$scheduler <- arg_value(args, "scheduler", "multi_step")
  cfg$sch_milestones <- parse_int_vec(arg_value(args, "sch-milestones", "1000,1400,1800"), c(1000L, 1400L, 1800L))
  cfg$sch_gamma <- as.numeric(arg_value(args, "sch-gamma", 0.5))
  cfg$penalty <- as.numeric(arg_value(args, "penalty", log(972)))
  cfg$epsilon_1 <- as.numeric(arg_value(args, "epsilon-1", 0.05))
  cfg$epsilon_T <- as.numeric(arg_value(args, "epsilon-T", 0.005))
  cfg$steps_T <- as.integer(arg_value(args, "steps-T", min(500L, cfg$epochs)))
  cfg$sic_threshold <- as.numeric(arg_value(args, "sic-threshold", 0.5))
  cfg$sic_threshold_type <- arg_value(args, "sic-threshold-type", "phi")
  cfg$post_refit_epochs <- as.integer(arg_value(args, "post-refit-epochs", 0L))
  cfg$post_refit_lr <- as.numeric(arg_value(args, "post-refit-lr", 0.01))
  cfg$standardize <- parse_logical(arg_value(args, "standardize", NULL), FALSE)
  cfg$seed <- as.integer(arg_value(args, "seed", 42L))
  cfg$workers <- as.integer(arg_value(args, "workers", 1L))
  cfg$torch_threads <- as.integer(arg_value(args, "torch-threads", 1L))
  cfg$show_epochs <- parse_logical(arg_value(args, "show-epochs", NULL), FALSE)
  cfg$data_dir <- arg_value(args, "data-dir", file.path("C:", "tmp", "ISLaB-LBBNN-JAIR", "data", "mice"))
  cfg$out <- arg_value(args, "out", file.path("uci_experiments", paste0("mice_islab_sicnn_", preset, "_results.rds")))

  if (!cfg$sic_threshold_type %in% c("phi", "abs")) stop("sic-threshold-type must be phi or abs")
  if (cfg$scheduler != "multi_step") stop("The Mice runner currently supports the LBBNN multi_step schedule only")
  cfg
}

main <- function() {
  cfg <- make_config(parse_args(commandArgs(trailingOnly = TRUE)))
  load_sicnn()
  torch::torch_set_num_threads(cfg$torch_threads)
  data_obj <- read_islab_mice_data(cfg$data_dir, cfg$standardize)
  cfg$n_train <- nrow(data_obj$x_train)
  cfg$n_test <- nrow(data_obj$x_test)
  cfg$workers <- max(1L, min(cfg$workers, cfg$reps))

  cat("ISLaB/LBBNN Mice Protein split with SICNN\n")
  cat(sprintf("preset=%s | reps=%d | n_train=%d | n_test=%d | p=%d | classes=%d\n",
              cfg$preset, cfg$reps, cfg$n_train, cfg$n_test, ncol(data_obj$x_train), data_obj$n_classes))
  cat(sprintf("architecture=%s | sigmoid | lr=%.3g | batch=%d | penalty=%.6g | epsilon %.3g -> %.3g over %d | refit=%d\n",
              paste(c(ncol(data_obj$x_train), cfg$hidden_sizes, data_obj$n_classes), collapse = "-"),
              cfg$lr, cfg$batch_size, cfg$penalty, cfg$epsilon_1, cfg$epsilon_T,
              cfg$steps_T, cfg$post_refit_epochs))

  started <- proc.time()[[3L]]
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
    parallel::clusterExport(cluster, varlist = setdiff(ls(envir = .GlobalEnv), "cfg"), envir = .GlobalEnv)
    parallel::clusterCall(cluster, setwd, getwd())
    chunks <- parallel::parLapplyLB(cluster, rep_ids, run_one_rep, cfg = cfg, data_obj = data_obj, total_reps = cfg$reps)
  }

  results <- do.call(rbind, chunks)
  attr(results, "config") <- cfg
  attr(results, "elapsed_seconds") <- proc.time()[[3L]] - started
  summary_table <- summarise_results(results)
  dir.create(dirname(cfg$out), recursive = TRUE, showWarnings = FALSE)
  saveRDS(results, cfg$out)
  write.csv(summary_table, sub("[.]rds$", "_summary.csv", cfg$out), row.names = FALSE)

  cat(sprintf("\nCompleted in %.1f minutes. Saved: %s\n", attr(results, "elapsed_seconds") / 60, cfg$out))
  print(summary_table)
}

main()