library(SICNN)
library(torch)
library(randomForest)
library(nnet)

# Helper to prepare datasets from local CSVs
prepare_uci_data <- function(name) {
  if (name == "Pima") {
    df <- read.csv("uci_experiments/pima.csv", header=FALSE)
    colnames(df) <- c("preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class")
    target_type <- "binary classification"
  } else if (name == "Sonar") {
    df <- read.csv("uci_experiments/sonar.csv", header=FALSE)
    target_idx <- ncol(df)
    df[[target_idx]] <- ifelse(df[[target_idx]] == "M", 1, 0)
    target_type <- "binary classification"
  } else if (name == "BreastCancer") {
    df <- read.table("uci_experiments/breast_cancer.csv", sep=",", header=FALSE, na.strings="?", quote="", row.names=NULL)
    df <- na.omit(df)
    df <- df[,-1] # remove ID
    # After removing ID, we have 10 columns. The last one (V11 originally) is target.
    target_idx <- ncol(df)
    # Class values: 2 for benign, 4 for malignant.
    df[[target_idx]] <- ifelse(as.numeric(df[[target_idx]]) == 4, 1, 0)
    for(i in seq_len(ncol(df))) df[[i]] <- as.numeric(as.character(df[[i]]))
    target_type <- "binary classification"
  } else if (name == "Wine") {
    # UCI Wine can have class in 1st or last column depending on source.
    # In uci_experiments/wine.csv it seems to be in the LAST column (14th).
    df <- read.csv("uci_experiments/wine.csv", header=FALSE)
    target_idx <- ncol(df)
    target <- df[[target_idx]]
    # Adjust to 0-indexed if needed
    if (min(target) == 1) {
      target <- target - 1
    }
    df <- df[,-target_idx]
    df$target <- target
    target_type <- "multiclass classification"
  } else if (name == "Airfoil") {
    df <- read.table("uci_experiments/airfoil.dat", header=FALSE)
    target_type <- "regression"
  } else if (name == "Yacht") {
    df <- read.table("uci_experiments/yacht.data", header=FALSE)
    target_type <- "regression"
  } else if (name == "Concrete") {
    df <- read.csv("uci_experiments/concrete_slump.csv", header=TRUE)
    df <- df[, 2:9] # Features 2:8, Target SLUMP at 9 (relative to original 1:11)
    target_type <- "regression"
  } else if (name == "Housing") {
    if (requireNamespace("mlbench", quietly = TRUE)) {
      data("BostonHousing", package = "mlbench")
      df <- mlbench::BostonHousing
      df$chas <- as.numeric(df$chas) - 1
      target_type <- "regression"
    } else {
      return(NULL)
    }
  }
  return(list(df=df, target_type=target_type))
}

run_experiment <- function(dataset_name, lr=0.01, penalty_mult=5, epochs=1000) {
  set.seed(42)
  cat(paste("\n--- Processing", dataset_name, "---"))
  data_obj <- tryCatch(prepare_uci_data(dataset_name), error=function(e) NULL)
  if (is.null(data_obj)) {
    cat(" Skip (not found or error)\n")
    return(NULL)
  }
  
  df <- data_obj$df
  type <- data_obj$target_type
  
  target_col <- ncol(df)
  features <- df[,-target_col]
  target <- df[[target_col]]
  
  # Normalize features
  if (type != "regression") {
    # Feature normalization for classification
    for(i in seq_len(ncol(features))) {
       r <- range(features[[i]], na.rm=TRUE)
       if (r[2] != r[1]) features[[i]] <- (features[[i]] - r[1]) / (r[2] - r[1])
    }
  } else {
    # regression: only feature normalization
    for(i in seq_len(ncol(features))) {
       r <- range(features[[i]], na.rm=TRUE)
       if (r[2] != r[1]) {
         features[[i]] <- (features[[i]] - r[1]) / (r[2] - r[1])
       } else {
         features[[i]] <- 0 # Constant feature
       }
    }
    r_y <- range(target)
    if (r_y[2] != r_y[1]) {
      target <- (target - r_y[1]) / (r_y[2] - r_y[1])
    }
  }
  
  df_norm <- cbind(features, target)
  colnames(df_norm)[ncol(df_norm)] <- "target"
  
  # DEBUG: print target range
  message(sprintf(" Target range: [%.3f, %.3f]\n", min(target, na.rm=TRUE), max(target, na.rm=TRUE)))
  
  n_total <- nrow(df_norm)
  n_train <- floor(n_total * 0.8)
  n_test <- n_total - n_train
  
  loaders <- get_dataloaders(df_norm, train_proportion = 0.8, train_batch_size = min(32, n_train), test_batch_size = n_test, standardize = FALSE)
  
  p <- ncol(features)
  if (type == "multiclass classification") {
    out_dim <- length(unique(target))
    sizes <- c(p, 10, 5, out_dim)
  } else {
    sizes <- c(p, 10, 5, 1)
  }
  
  model <- SICNN_Net(problem_type = type, sizes = sizes, input_skip = TRUE, device = "cpu")
  
  # Train
  train_SICNN(
    epochs = epochs, restarts = 1, SICNN = model, 
    lr = lr, train_dl = loaders$train_loader, device = "cpu",
    scheduler = "step", sch_step_size = floor(epochs/3), n_train = n_train,
    epsilon_1 = 1, epsilon_T = 1e-4, steps_T = floor(epochs*0.8), 
    sic_threshold = 0.5, penalty = penalty_mult * log(n_train)
  )
  
  # Final validation on test set
  val <- validate_SICNN(SICNN = model, num_samples = 1, test_dl = loaders$test_loader, device="cpu")
  
  # Data for metrics
  train_idx <- 1:floor(nrow(df_norm) * 0.8)
  train_df <- df_norm[train_idx, ]
  test_df <- df_norm[-train_idx, ]

  if (type == "regression") {
     # Calculate R2 for the sparse model manually
     SIC_pred <- as.numeric(model(torch::torch_tensor(as.matrix(features[-train_idx, ]), dtype=torch::torch_float()), sparse=TRUE))
     rss_sic <- sum((SIC_pred - test_df$target)^2, na.rm=TRUE)
     tss_sic <- sum((test_df$target - mean(test_df$target, na.rm=TRUE))^2, na.rm=TRUE)
     metric_val <- 1 - (rss_sic / tss_sic)
  } else {
     metric_val <- val$accuracy_sparse
  }
  
  # --- Baselines ---
  message(" Running Baselines...")
  set.seed(42)
  
  lm_metric <- NA
  rf_metric <- NA
  
  tryCatch({
    if (type == "regression") {
      lm_fit <- lm(target ~ ., data = train_df)
      lm_pred <- predict(lm_fit, test_df)
      rss <- sum((lm_pred - test_df$target)^2, na.rm=TRUE)
      tss <- sum((test_df$target - mean(test_df$target, na.rm=TRUE))^2, na.rm=TRUE)
      lm_metric <- 1 - (rss / tss)
    } else if (type == "binary classification") {
      glm_fit <- glm(target ~ ., data = train_df, family = binomial)
      glm_prob <- predict(glm_fit, test_df, type = "response")
      glm_pred <- ifelse(glm_prob > 0.5, 1, 0)
      lm_metric <- mean(glm_pred == test_df$target)
    } else {
      mn_fit <- nnet::multinom(target ~ ., data = train_df, trace=FALSE)
      mn_pred <- predict(mn_fit, test_df)
      lm_metric <- mean(as.numeric(as.character(mn_pred)) == test_df$target)
    }

    if (type == "regression") {
      rf_fit <- randomForest::randomForest(target ~ ., data = train_df)
      rf_pred <- predict(rf_fit, test_df)
      rss_rf <- sum((rf_pred - test_df$target)^2, na.rm=TRUE)
      tss_rf <- sum((test_df$target - mean(test_df$target, na.rm=TRUE))^2, na.rm=TRUE)
      rf_metric <- 1 - (rss_rf / tss_rf)
    } else {
      train_df$target <- as.factor(train_df$target)
      rf_fit <- randomForest::randomForest(target ~ ., data = train_df)
      rf_pred <- predict(rf_fit, test_df)
      rf_metric <- mean(as.numeric(as.character(rf_pred)) == test_df$target)
    }
  }, error = function(e) {
    message(paste("\n Baseline error:", e$message))
  })
  
  cat(sprintf("\n SICNN: %.3f | GLM: %.3f | RF: %.3f (Sparsity: %.2f%%)\n", metric_val, lm_metric, rf_metric, val$sparsity_pct))
  
  return(list(name=dataset_name, metric=metric_val, lm_metric=lm_metric, rf_metric=rf_metric, sparsity=val$sparsity_pct, p=p, type=type))
}

args <- commandArgs(trailingOnly=TRUE)
datasets <- c("Pima", "Sonar", "BreastCancer", "Wine", "Airfoil", "Yacht", "Concrete")

if (length(args) > 0) {
  target_datasets <- args
} else {
  target_datasets <- datasets
}

results <- list()
for (d in target_datasets) {
  p_mult <- switch(d,
    "Pima" = 0.5,
    "Sonar" = 1.0,
    "BreastCancer" = 5.0,
    "Wine" = 1.0,
    "Airfoil" = 0.1,
    "Yacht" = 0.1,
    "Concrete" = 0.1,
    5.0
  )
  res <- tryCatch(run_experiment(d, penalty_mult = p_mult), error=function(e) { message(paste("Experiment failed for", d, ":", e$message)); NULL })
  results[[d]] <- res
}

if (length(args) == 0) {
  cat("\n\n--- UCI BENCHMARK SUMMARY (incl. Baselines) ---\n")
  cat(sprintf("%-15s | %-10s | %-7s | %-7s | %-7s | %-10s\n", "Dataset", "Type", "SICNN", "GLM", "RF", "Sparsity"))
  cat("--------------------------------------------------------------------------------\n")
  for (d in datasets) {
    res <- results[[d]]
    if (!is.null(res)) {
      cat(sprintf("%-15s | %-10s | %-7.3f | %-7.3f | %-7.3f | %-10.2f%%\n", res$name, res$type, res$metric, res$lm_metric, res$rf_metric, res$sparsity))
    } else {
      cat(sprintf("%-15s | %-10s | %-7s | %-7s | %-7s | %-10s\n", d, "N/A", "N/A", "N/A", "N/A", "N/A"))
    }
  }
}
