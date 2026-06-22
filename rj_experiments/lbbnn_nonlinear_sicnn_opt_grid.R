#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(torch)
  library(tibble)
  library(dplyr)
})

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    out[[parts[[1L]]]] <- if (length(parts) == 1L) TRUE else paste(parts[-1L], collapse = "=")
  }
  out
}

arg_value <- function(args, name, default = NULL) if (!is.null(args[[name]])) args[[name]] else default
parse_num_vec <- function(x, default) if (is.null(x)) default else as.numeric(strsplit(x, ",", fixed = TRUE)[[1L]])
parse_chr_vec <- function(x, default) if (is.null(x)) default else strsplit(x, ",", fixed = TRUE)[[1L]]
parse_logical <- function(x, default) {
  if (is.null(x)) return(default)
  value <- tolower(as.character(x))
  if (!value %in% c("true", "false", "1", "0", "yes", "no")) stop("Expected a logical value, got: ", x)
  value %in% c("true", "1", "yes")
}

load_sicnn <- function() {
  if (requireNamespace("devtools", quietly = TRUE) && file.exists("DESCRIPTION")) {
    suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
  } else {
    suppressPackageStartupMessages(library(SICNN))
  }
}

make_lbbnn_nonlinear_data <- function(n_train, n_test, rho, noise_sd, seed) {
  set.seed(seed)
  n_total <- n_train + n_test
  x <- matrix(stats::runif(n_total * 4L, min = -10, max = 10), ncol = 4L)
  x[, 3L] <- rho * x[, 1L] + (1 - rho) * x[, 3L]
  colnames(x) <- paste0("x", seq_len(4L))
  eta <- 100 + x[, 1L] + x[, 2L] + x[, 1L] * x[, 2L] + x[, 1L]^2 + x[, 2L]^2 + stats::rnorm(n_total, sd = noise_sd)
  data <- as.data.frame(x)
  data$y <- as.numeric(eta >= stats::median(eta))
  list(
    train = data[seq_len(n_train), , drop = FALSE],
    test = data[n_train + seq_len(n_test), , drop = FALSE],
    true_active = c(TRUE, TRUE, FALSE, FALSE)
  )
}

load_lbbnn_nonlinear_data <- function(data_dir, rho) {
  rho_tag <- formatC(rho, format = "f", digits = 1L)
  read_matrix <- function(name) as.matrix(utils::read.table(file.path(data_dir, paste0(rho_tag, name)), sep = ",", header = FALSE))
  x_train <- read_matrix("X_train.txt")
  x_test <- read_matrix("X_test.txt")
  y_train <- as.numeric(read_matrix("Y_train.txt"))
  y_test <- as.numeric(read_matrix("Y_test.txt"))
  if (ncol(x_train) != 4L || ncol(x_test) != 4L) stop("Expected four covariates in LBBNN nonlinear data")
  names_train <- paste0("x", seq_len(4L))
  train <- as.data.frame(x_train); test <- as.data.frame(x_test)
  names(train) <- names_train; names(test) <- names_train
  train$y <- y_train; test$y <- y_test
  list(train = train, test = test, true_active = c(TRUE, TRUE, FALSE, FALSE))
}
standardize_sim_inputs <- function(sim) {
  features <- paste0("x", seq_len(4L))
  x_train <- as.matrix(sim$train[, features, drop = FALSE])
  center <- colMeans(x_train)
  scale <- apply(x_train, 2L, stats::sd)
  scale[!is.finite(scale) | scale == 0] <- 1
  for (split_name in c("train", "test")) {
    x <- as.matrix(sim[[split_name]][, features, drop = FALSE])
    sim[[split_name]][, features] <- sweep(sweep(x, 2L, center, "-"), 2L, scale, "/")
  }
  list(sim = sim, center = center, scale = scale)
}

make_loader <- function(data, batch_size, shuffle) {
  ds <- torch::tensor_dataset(
    torch::torch_tensor(as.matrix(data[, paste0("x", seq_len(4L)), drop = FALSE]), dtype = torch::torch_float()),
    torch::torch_tensor(as.numeric(data$y), dtype = torch::torch_float())
  )
  torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)
}

make_activation <- function(name) {
  switch(name,
    sigmoid = torch::nn_sigmoid(),
    relu = torch::nn_relu(),
    leaky_relu = torch::nn_leaky_relu(0),
    stop("Unknown activation: ", name)
  )
}

scale_layer_columns <- function(layer, scales, device) {
  scale_tensor <- torch::torch_tensor(
    matrix(scales, nrow = layer$out_features, ncol = layer$in_features, byrow = TRUE),
    dtype = torch::torch_float(), device = device
  )
  layer$weight_mean$data()$mul_(scale_tensor)
}

apply_structured_init <- function(model, mode, p, hidden_scale, direct_scale, covariate_scale, device) {
  if (mode == "default") return(invisible(NULL))
  if (!mode %in% c("direct_skip", "lbbnn_like")) stop("Unknown init mode: ", mode)
  for (layer in model$layers$children) {
    scales <- rep(hidden_scale, layer$in_features)
    if (mode == "lbbnn_like") {
      cov_cols <- if (layer$in_features == p) seq_len(p) else (layer$in_features - p + 1L):layer$in_features
      scales[cov_cols] <- covariate_scale
    }
    scale_layer_columns(layer, scales, device)
  }
  scales <- rep(hidden_scale, model$out_layer$in_features)
  cov_cols <- if (model$out_layer$in_features == p) seq_len(p) else (model$out_layer$in_features - p + 1L):model$out_layer$in_features
  scales[cov_cols] <- direct_scale
  scale_layer_columns(model$out_layer, scales, device)
  invisible(NULL)
}

compute_active_paths <- function(model, cfg) {
  if (isTRUE(model$input_skip)) {
    model$compute_paths_input_skip(epsilon = cfg$epsilon_T, threshold = cfg$sic_threshold, threshold_type = cfg$sic_threshold_type)
  } else {
    model$compute_paths(epsilon = cfg$epsilon_T, threshold = cfg$sic_threshold, threshold_type = cfg$sic_threshold_type)
  }
}

select_sicnn_features <- function(model, p, cfg) {
  compute_active_paths(model, cfg)
  selected <- rep(FALSE, p)
  layers <- c(unname(model$layers$children), list(model$out_layer))
  for (layer in layers) {
    alpha <- as.matrix(layer$alpha_active_path$cpu())
    cov_cols <- if (ncol(alpha) == p) seq_len(p) else (ncol(alpha) - p + 1L):ncol(alpha)
    selected <- selected | colSums(alpha[, cov_cols, drop = FALSE]) > 0
  }
  selected
}

direct_output_active <- function(model, p, cfg) {
  if (!isTRUE(model$input_skip)) return(rep(FALSE, p))
  w <- as.matrix(model$out_layer$weight_mean$detach()$cpu())
  w <- w[, (ncol(w) - p + 1L):ncol(w), drop = FALSE]
  if (cfg$sic_threshold_type == "abs") return(as.logical(abs(w[1L, ]) > cfg$sic_threshold))
  phi <- w^2 / (w^2 + cfg$epsilon_T^2)
  as.logical(phi[1L, ] > cfg$sic_threshold)
}

predict_binary_probs <- function(model, loader, sparse) {
  model$eval()
  probability <- numeric(0)
  torch::with_no_grad({
    coro::loop(for (batch in loader) {
      probability <- c(probability, as.numeric(model(batch[[1]], sparse = sparse)$squeeze()$cpu()))
    })
  })
  probability
}

binary_metrics <- function(y, probability) {
  probability <- pmin(pmax(probability, 1e-8), 1 - 1e-8)
  list(
    acc = mean(as.numeric(probability >= 0.5) == y),
    nll = -mean(y * log(probability) + (1 - y) * log(1 - probability))
  )
}

make_epsilon_by_epoch <- function(epochs, epsilon_1, epsilon_T, steps_T, schedule, power) {
  epsilon_seq <- epsilon_1 * (epsilon_T / epsilon_1)^((0:(steps_T - 1L)) / max(1L, steps_T - 1L))
  t <- seq_len(epochs) / epochs
  fraction <- switch(schedule,
    frontloaded = t^power,
    backloaded = 1 - (1 - t)^power,
    stop("Unknown epsilon schedule: ", schedule)
  )
  epsilon_seq[pmin(steps_T, pmax(1L, ceiling(fraction * steps_T)))]
}

train_sicnn_custom_epsilon <- function(epochs, model, lr, train_loader, device, scheduler_mode, sch_step_size, n_train,
                                       epsilon_1, epsilon_T, steps_T, epsilon_schedule, schedule_power,
                                       sic_threshold, sic_threshold_type, penalty) {
  epsilon_by_epoch <- make_epsilon_by_epoch(epochs, epsilon_1, epsilon_T, steps_T, epsilon_schedule, schedule_power)
  model$sic_epsilon_T <- epsilon_T
  model$sic_threshold <- sic_threshold
  model$sic_report_threshold_type <- sic_threshold_type
  model$sic_penalty <- penalty
  model$n_train <- n_train
  optimizer <- torch::optim_adam(model$parameters, lr = lr)
  scheduler <- if (scheduler_mode == "none") NULL else torch::lr_step(optimizer, step_size = sch_step_size, gamma = 0.1)
  history <- list(loss = numeric(epochs), accs = numeric(epochs), active_weights = numeric(epochs))
  for (epoch in seq_len(epochs)) {
    model$train()
    correct <- 0
    total <- 0
    losses <- numeric(0)
    epsilon <- epsilon_by_epoch[[epoch]]
    report_smooth_count <- model$smooth_param_count(epsilon_T)$detach()$item()
    coro::loop(for (batch in train_loader) {
      optimizer$zero_grad()
      x <- batch[[1]]$to(device = device)
      y <- batch[[2]]$to(device = device)
      probability <- model(x, sparse = FALSE)$squeeze()
      data_loss <- model$loss_fn(probability, y)
      batch_n <- dim(x)[1L]
      loss <- 2 * (n_train / batch_n) * data_loss + penalty * model$smooth_param_count(epsilon)
      losses <- c(losses, 2 * (n_train / batch_n) * data_loss$item() + penalty * report_smooth_count)
      correct <- correct + sum((probability > 0.5) == y)$item()
      total <- total + length(y)
      loss$backward()
      optimizer$step()
    })
    if (!is.null(scheduler)) scheduler$step()
    counts <- model$sic_weight_counts(epsilon = epsilon_T, threshold = sic_threshold, threshold_type = sic_threshold_type, active_paths = FALSE)
    history$loss[[epoch]] <- mean(losses)
    history$accs[[epoch]] <- correct / total
    history$active_weights[[epoch]] <- as.numeric(counts[["active"]])
  }
  history
}

fit_one <- function(job, cfg, sim, train_loader, test_loader) {
  set.seed(job$seed)
  torch::torch_manual_seed(job$seed)
  model <- SICNN_Net(
    problem_type = "binary classification", sizes = c(4L, cfg$hidden_sizes, 1L),
    input_skip = cfg$input_skip, device = cfg$device, custom_act = make_activation(cfg$activation)
  )
  apply_structured_init(model, job$init_mode, 4L, job$hidden_init_scale, job$direct_init_scale, job$covariate_init_scale, cfg$device)
  penalty <- job$penalty_mult * log(cfg$n_train)
  scheduler <- if (job$scheduler_mode == "none") NULL else "step"
  sch_step_size <- switch(job$scheduler_mode, none = NULL, mid = max(1L, floor(cfg$epochs * 0.5)), late = max(1L, floor(cfg$epochs * 0.75)))

  if (cfg$warmup_epochs > 0L) {
    suppressMessages(train_SICNN(
      epochs = cfg$warmup_epochs, restarts = 1L, SICNN = model, lr = job$lr, train_dl = train_loader,
      device = cfg$device, scheduler = NULL, n_train = cfg$n_train,
      epsilon_1 = cfg$epsilon_1, epsilon_T = cfg$epsilon_1, steps_T = 1L,
      sic_threshold = cfg$sic_threshold, sic_threshold_type = cfg$sic_threshold_type,
      penalty = cfg$warmup_penalty_mult * log(cfg$n_train)
    ))
  }
  warmup_metrics <- if (cfg$warmup_epochs > 0L) binary_metrics(sim$test$y, predict_binary_probs(model, test_loader, sparse = FALSE)) else list(acc = NA_real_, nll = NA_real_)

  start <- proc.time()[[3L]]
  history <- if (cfg$epsilon_schedule == "linear") {
    suppressMessages(train_SICNN(
      epochs = cfg$epochs, restarts = 1L, SICNN = model, lr = job$lr, train_dl = train_loader,
      device = cfg$device, scheduler = scheduler, sch_step_size = sch_step_size, n_train = cfg$n_train,
      epsilon_1 = cfg$epsilon_1, epsilon_T = cfg$epsilon_T, steps_T = cfg$steps_T,
      sic_threshold = cfg$sic_threshold, sic_threshold_type = cfg$sic_threshold_type, penalty = penalty
    ))
  } else {
    train_sicnn_custom_epsilon(
      cfg$epochs, model, job$lr, train_loader, cfg$device, job$scheduler_mode, sch_step_size, cfg$n_train,
      cfg$epsilon_1, cfg$epsilon_T, cfg$steps_T, cfg$epsilon_schedule, cfg$schedule_power,
      cfg$sic_threshold, cfg$sic_threshold_type, penalty
    )
  }
  elapsed <- proc.time()[[3L]] - start
  dense <- binary_metrics(sim$test$y, predict_binary_probs(model, test_loader, sparse = FALSE))

  bind_rows(lapply(cfg$report_thresholds, function(report_threshold) {
    report_cfg <- cfg
    report_cfg$sic_threshold <- report_threshold
    compute_active_paths(model, report_cfg)
    sparse <- binary_metrics(sim$test$y, predict_binary_probs(model, test_loader, sparse = TRUE))
    selected <- select_sicnn_features(model, 4L, report_cfg)
    direct_active <- direct_output_active(model, 4L, report_cfg)
    counts <- model$sic_weight_counts(epsilon = cfg$epsilon_T, threshold = report_threshold, threshold_type = cfg$sic_threshold_type, active_paths = TRUE)
    tp <- sum(selected & sim$true_active)
    fp <- sum(selected & !sim$true_active)
    fn <- sum(!selected & sim$true_active)
    sparse_goal <- as.numeric(counts[["active"]]) <= cfg$max_used_weights && identical(selected, sim$true_active) && sparse$acc >= cfg$min_sparse_acc
    tibble(
      job_id = job$job_id, rho = cfg$rho, n_train = cfg$n_train, n_test = cfg$n_test, data_source = ifelse(is.null(cfg$data_dir), "generated", "lbbnn_saved"),
      standardize_inputs = cfg$standardize_inputs, warmup_epochs = cfg$warmup_epochs,
      warmup_penalty_mult = cfg$warmup_penalty_mult, warmup_acc_full = warmup_metrics$acc, warmup_nll_full = warmup_metrics$nll,
      epochs = cfg$epochs, lr = job$lr, scheduler_mode = job$scheduler_mode, sch_step_size = ifelse(is.null(sch_step_size), NA_integer_, sch_step_size),
      penalty_mult = job$penalty_mult, penalty = penalty, init_mode = job$init_mode,
      epsilon_1 = cfg$epsilon_1, epsilon_T = cfg$epsilon_T, steps_T = cfg$steps_T, epsilon_schedule = cfg$epsilon_schedule, schedule_power = cfg$schedule_power,
      report_threshold = report_threshold, threshold_type = cfg$sic_threshold_type,
      hidden_init_scale = job$hidden_init_scale, covariate_init_scale = job$covariate_init_scale, direct_init_scale = job$direct_init_scale,
      acc_full = dense$acc, acc_sparse = sparse$acc, nll_full = dense$nll, nll_sparse = sparse$nll,
      used_weights = as.numeric(counts[["active"]]), total_weights = as.numeric(counts[["total"]]),
      avg_depth = NA_real_, max_depth = NA_integer_, n_active_paths = NA_integer_, selected_count = sum(selected),
      tp = tp, fp = fp, fn = fn, exact_support = identical(selected, sim$true_active),
      direct_x1 = direct_active[[1L]], direct_x2 = direct_active[[2L]], direct_x3 = direct_active[[3L]], direct_x4 = direct_active[[4L]],
      selected_x1 = selected[[1L]], selected_x2 = selected[[2L]], selected_x3 = selected[[3L]], selected_x4 = selected[[4L]],
      sparse_goal = sparse_goal, final_train_loss = tail(history$loss, 1L), final_train_acc = tail(history$accs, 1L),
      final_train_active_weights = tail(history$active_weights, 1L), elapsed_seconds = elapsed
    )
  }))
}

make_config <- function(args) {
  cfg <- list(
    n_train = 1000L, n_test = 1000L, rho = 0, noise_sd = 0.01, seed = 20260618L, data_dir = NULL,
    hidden_sizes = c(20L, 20L, 20L, 20L), epochs = 60L, warmup_epochs = 0L, warmup_penalty_mult = 1e-8,
    iter_per_epoch = 10L, activation = "sigmoid", input_skip = TRUE, standardize_inputs = FALSE,
    lrs = c(0.005), scheduler_modes = c("late"), init_modes = c("lbbnn_like"), penalty_mults = c(1),
    epsilon_1 = 0.1, epsilon_T = 0.005, steps_T = 60L, epsilon_schedule = "linear", schedule_power = 3,
    sic_threshold = 0.5, sic_threshold_type = "phi", report_thresholds = 0.5,
    hidden_init_scales = 0.5, covariate_init_scales = 1, direct_init_scales = 1,
    torch_threads = 1L, device = "cpu", min_sparse_acc = 0.95, max_used_weights = 200L,
    out = file.path("rj_experiments", "optimizer_sweep", "lbbnn_nonlinear_sicnn_opt_grid_results.rds")
  )
  cfg$n_train <- as.integer(arg_value(args, "n-train", cfg$n_train)); cfg$n_test <- as.integer(arg_value(args, "n-test", cfg$n_test))
  cfg$rho <- as.numeric(arg_value(args, "rho", cfg$rho)); cfg$noise_sd <- as.numeric(arg_value(args, "noise-sd", cfg$noise_sd)); cfg$seed <- as.integer(arg_value(args, "seed", cfg$seed)); cfg$data_dir <- arg_value(args, "data-dir", cfg$data_dir)
  cfg$epochs <- as.integer(arg_value(args, "epochs", cfg$epochs)); cfg$warmup_epochs <- as.integer(arg_value(args, "warmup-epochs", cfg$warmup_epochs)); cfg$warmup_penalty_mult <- as.numeric(arg_value(args, "warmup-penalty-mult", cfg$warmup_penalty_mult))
  cfg$iter_per_epoch <- as.integer(arg_value(args, "iter-per-epoch", cfg$iter_per_epoch)); cfg$activation <- arg_value(args, "activation", cfg$activation)
  cfg$input_skip <- parse_logical(arg_value(args, "input-skip"), cfg$input_skip); cfg$standardize_inputs <- parse_logical(arg_value(args, "standardize-inputs"), cfg$standardize_inputs)
  cfg$lrs <- parse_num_vec(arg_value(args, "lrs"), cfg$lrs); cfg$scheduler_modes <- parse_chr_vec(arg_value(args, "schedulers"), cfg$scheduler_modes); cfg$init_modes <- parse_chr_vec(arg_value(args, "init-modes"), cfg$init_modes)
  cfg$penalty_mults <- parse_num_vec(arg_value(args, "penalty-mults"), cfg$penalty_mults); cfg$epsilon_1 <- as.numeric(arg_value(args, "epsilon-1", cfg$epsilon_1)); cfg$epsilon_T <- as.numeric(arg_value(args, "epsilon-T", cfg$epsilon_T)); cfg$steps_T <- as.integer(arg_value(args, "steps-T", cfg$steps_T))
  cfg$epsilon_schedule <- arg_value(args, "epsilon-schedule", cfg$epsilon_schedule); cfg$schedule_power <- as.numeric(arg_value(args, "schedule-power", cfg$schedule_power)); cfg$sic_threshold <- as.numeric(arg_value(args, "sic-threshold", cfg$sic_threshold)); cfg$sic_threshold_type <- arg_value(args, "sic-threshold-type", cfg$sic_threshold_type)
  cfg$report_thresholds <- parse_num_vec(arg_value(args, "report-thresholds"), cfg$report_thresholds); cfg$hidden_init_scales <- parse_num_vec(arg_value(args, "hidden-init-scales"), cfg$hidden_init_scales); cfg$covariate_init_scales <- parse_num_vec(arg_value(args, "covariate-init-scales"), cfg$covariate_init_scales); cfg$direct_init_scales <- parse_num_vec(arg_value(args, "direct-init-scales"), cfg$direct_init_scales)
  cfg$torch_threads <- as.integer(arg_value(args, "torch-threads", cfg$torch_threads)); cfg$min_sparse_acc <- as.numeric(arg_value(args, "min-sparse-acc", cfg$min_sparse_acc)); cfg$max_used_weights <- as.integer(arg_value(args, "max-used-weights", cfg$max_used_weights)); cfg$out <- arg_value(args, "out", cfg$out)
  if (!all(cfg$scheduler_modes %in% c("none", "mid", "late"))) stop("Schedulers must be among: none, mid, late")
  if (!all(cfg$init_modes %in% c("default", "direct_skip", "lbbnn_like"))) stop("Unknown init mode")
  if (!cfg$epsilon_schedule %in% c("linear", "frontloaded", "backloaded")) stop("Unknown epsilon schedule")
  if (any(!is.finite(cfg$report_thresholds)) || any(cfg$report_thresholds <= 0) || (cfg$sic_threshold_type == "phi" && any(cfg$report_thresholds >= 1))) stop("Invalid report-thresholds")
  cfg$batch_size <- max(1L, ceiling(cfg$n_train / cfg$iter_per_epoch)); cfg$test_batch_size <- cfg$n_test
  cfg
}

main <- function() {
  cfg <- make_config(parse_args(commandArgs(trailingOnly = TRUE)))
  load_sicnn(); torch::torch_set_num_threads(cfg$torch_threads)
  sim <- if (is.null(cfg$data_dir)) {
    make_lbbnn_nonlinear_data(cfg$n_train, cfg$n_test, cfg$rho, cfg$noise_sd, cfg$seed)
  } else {
    load_lbbnn_nonlinear_data(cfg$data_dir, cfg$rho)
  }
  cfg$n_train <- nrow(sim$train)
  cfg$n_test <- nrow(sim$test)
  if (cfg$standardize_inputs) sim <- standardize_sim_inputs(sim)$sim
  train_loader <- make_loader(sim$train, cfg$batch_size, TRUE); test_loader <- make_loader(sim$test, cfg$test_batch_size, FALSE)
  jobs <- expand.grid(lr = cfg$lrs, scheduler_mode = cfg$scheduler_modes, init_mode = cfg$init_modes, penalty_mult = cfg$penalty_mults, hidden_init_scale = cfg$hidden_init_scales, covariate_init_scale = cfg$covariate_init_scales, direct_init_scale = cfg$direct_init_scales, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  jobs$job_id <- seq_len(nrow(jobs)); jobs$seed <- cfg$seed + 1000L + jobs$job_id
  dir.create(dirname(cfg$out), recursive = TRUE, showWarnings = FALSE)
  results <- vector("list", nrow(jobs))
  for (i in seq_len(nrow(jobs))) {
    job <- as.list(jobs[i, ])
    cat(sprintf("Job %d/%d: penalty=%g*log(n), warmup=%d, thresholds=%s\n", i, nrow(jobs), job$penalty_mult, cfg$warmup_epochs, paste(cfg$report_thresholds, collapse = ",")))
    results[[i]] <- fit_one(job, cfg, sim, train_loader, test_loader)
    partial <- bind_rows(results[seq_len(i)]); attr(partial, "config") <- cfg; saveRDS(partial, cfg$out)
    flush.console()
  }
  out <- bind_rows(results); attr(out, "config") <- cfg; saveRDS(out, cfg$out)
  ranked <- out |> mutate(rank_score = 1000 * as.integer(!sparse_goal) + 100 * as.integer(!exact_support) + 10 * pmax(0, cfg$min_sparse_acc - acc_sparse) + pmax(0, used_weights - cfg$max_used_weights) + used_weights / 1000) |> arrange(rank_score, desc(acc_sparse), used_weights)
  summary <- out |> group_by(penalty_mult, report_threshold, warmup_epochs) |> summarise(acc_sparse = mean(acc_sparse), nll_sparse = mean(nll_sparse), used_weights = mean(used_weights), exact_support = mean(exact_support), sparse_goal = mean(sparse_goal), .groups = "drop") |> arrange(desc(sparse_goal), desc(exact_support), desc(acc_sparse))
  write.csv(out, sub("[.]rds$", "_raw.csv", cfg$out), row.names = FALSE); write.csv(ranked, sub("[.]rds$", "_ranked.csv", cfg$out), row.names = FALSE); write.csv(summary, sub("[.]rds$", "_summary.csv", cfg$out), row.names = FALSE)
  print(as.data.frame(summary))
}

main()