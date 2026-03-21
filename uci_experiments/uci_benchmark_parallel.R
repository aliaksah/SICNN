library(SICNN)
library(torch)
library(parallel)

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
  } else if (name == "Wine") {
    df <- read.csv("uci_experiments/wine.csv", header=FALSE)
    target <- df[[1]] - 1
    df <- df[,-1]
    df$target <- target
    target_type <- "multiclass classification"
  } else if (name == "BreastCancer") {
    df <- read.csv("uci_experiments/breast_cancer.csv", header=FALSE)
    df[df == "?"] <- NA
    df <- na.omit(df)
    df <- df[,-1]
    target_idx <- ncol(df)
    df[[target_idx]] <- ifelse(df[[target_idx]] == 4, 1, 0)
    for(i in 1:ncol(df)) df[[i]] <- as.numeric(as.character(df[[i]]))
    target_type <- "binary classification"
  }
  return(list(df=df, target_type=target_type))
}

run_experiment <- function(dataset_name, lr=0.01, penalty_mult=15, epochs=4000) {
  set.seed(42)
  cat(paste("Starting", dataset_name, "...\n"))
  data_obj <- tryCatch(prepare_uci_data(dataset_name), error=function(e) NULL)
  if (is.null(data_obj)) return(NULL)
  
  df <- data_obj$df
  type <- data_obj$target_type
  
  target_col <- ncol(df)
  features <- df[,-target_col]
  target <- df[[target_col]]
  
  for(i in 1:ncol(features)) {
     r <- range(features[[i]], na.rm=TRUE)
     if (r[2] != r[1]) features[[i]] <- (features[[i]] - r[1]) / (r[2] - r[1])
  }
  df_norm <- cbind(features, target)
  colnames(df_norm)[ncol(df_norm)] <- "target"
  
  loaders <- get_dataloaders(df_norm, train_proportion = 0.8, train_batch_size = min(64, nrow(df_norm)*0.8), test_batch_size = nrow(df_norm))
  
  p <- ncol(features)
  if (type == "multiclass classification") {
    out_dim <- length(unique(target))
    sizes <- c(p, 10, 5, out_dim)
  } else {
    sizes <- c(p, 10, 5, 1)
  }
  
  model <- SICNN_Net(problem_type = type, sizes = sizes, input_skip = TRUE, device = "cpu")
  
  suppressMessages(train_SICNN(
    epochs = epochs, restarts = 2, SICNN = model, 
    lr = lr, train_dl = loaders$train_loader, device = "cpu",
    scheduler = "step", sch_step_size = floor(epochs/3), n_train = nrow(df_norm)*0.8,
    epsilon_1 = 1, epsilon_T = 1e-4, steps_T = floor(epochs*0.8), 
    sic_threshold = 0.5, penalty = penalty_mult * log(nrow(df_norm)*0.8),
    verbose = FALSE
  ))
  
  val <- validate_SICNN(SICNN = model, num_samples = 1, test_dl = loaders$test_loader, device="cpu")
  cat(paste("Finished", dataset_name, "\n"))
  return(list(name=dataset_name, acc=val$accuracy_sparse, sparsity=val$sparsity_pct, p=p))
}

datasets <- c("Pima", "Sonar", "BreastCancer", "Wine")

# Run in parallel using mclapply (Mac/Linux)
results <- mclapply(datasets, run_experiment, mc.cores = length(datasets))

cat("\n--- PARALLEL UCI BENCHMARK SUMMARY ---\n")
cat(sprintf("%-15s | %-10s | %-15s | %-10s\n", "Dataset", "Features", "Accuracy", "Sparsity"))
cat("------------------------------------------------------------\n")
for (res in results) {
  if (!is.null(res)) {
    cat(sprintf("%-15s | %-10d | %-15.3f | %-10.2f%%\n", res$name, res$p, res$acc, res$sparsity))
  }
}
