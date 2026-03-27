library(Matrix)
require(graphics)

#' @title Function that checks how many times inputs are included, and from which layer. Used in summary function.
#' @description Useful when the number of inputs and/or hidden neurons are very
#' large, and direct visualization of the network is difficult.
#' @param model An instance of \code{SICNN_Net} where \code{input_skip = TRUE}.
#' @return A matrix of shape (p, L-1) where p is the number of input variables
#' and L the total number of layers (including input and output), with each element being 1 if the variable is included
#' or 0 if not included.
#' @keywords internal
get_input_inclusions <- function(model) {
  if (model$input_skip == FALSE) (stop("This function is currently only implemented for input-skip"))
  x_names <- c()
  layer_names <- c()
  for (k in 1:model$sizes[1]) {
    x_names <- c(x_names, paste("x", k - 1, sep = ""))
  }
  for (l in 1:(length(model$sizes) - 1)) {
    layer_names <- c(layer_names, paste("L", l - 1, sep = ""))
  }


  inclusion_matrix <- matrix(0, nrow = model$sizes[1], ncol = length(model$sizes) - 1)
  # add the names
  colnames(inclusion_matrix) <- layer_names
  rownames(inclusion_matrix) <- x_names


  inp_size <- model$sizes[1]
  i <- 1
  for (l in model$layers$children) {
    alp <- l$alpha_active_path
    incl <- as.numeric(torch::torch_sum(alp[, -inp_size:dim(alp)[2]], dim = 1))
    inclusion_matrix[, i] <- incl
    i <- i + 1
  }
  alp_out <- model$out_layer$alpha_active_path
  incl <- as.numeric(torch::torch_sum(alp_out[, -inp_size:dim(alp_out)[2]], dim = 1))
  inclusion_matrix[, i] <- incl
  i <- i + 1

  return(inclusion_matrix)
}


#' @title Summary of SICNN fit
#' @description Summary method for objects of the \code{SICNN_Net} class.
#' Only applies to objects trained with \code{input_skip = TRUE}.
#' @param object An object of class \code{SICNN_Net}.
#' @param epsilon numeric, the final epsilon parameter corresponding to the trained model. Default is 1e-5.
#' @param threshold numeric, the threshold for determining active edges. Default is 0.5.
#' @param ... further arguments passed to or from other methods.
#' @details
#' The returned table combines two types of information:
#' \itemize{
#'   \item Number of times each input variable is included in the active paths from each layer
#'         (obtained from \code{get_input_inclusions()}).
#'   \item Average inclusion probabilities for each input variable from each layer, including a final
#'         column showing the average across all layers.
#' }
#' @return A \code{data.frame} containing the above information. The function prints a formatted summary to the console.
#'   The returned \code{data.frame} is invisible.
#' @examples
#' \donttest{
#' x <- torch::torch_randn(3, 2)
#' b <- torch::torch_rand(2)
#' y <- torch::torch_matmul(x, b)
#' train_data <- torch::tensor_dataset(x, y)
#' train_loader <- torch::dataloader(train_data, batch_size = 3, shuffle = FALSE)
#' problem <- "regression"
#' sizes <- c(2, 1, 1)
#' model <- SICNN_Net(problem, sizes, input_skip = TRUE)
#' train_SICNN(epochs = 1, SICNN = model, lr = 0.01, train_dl = train_loader, n_train = 3)
#' summary(model)
#' }
#' @export
summary.SICNN_Net <- function(object,
                              epsilon = 1e-5,
                              threshold = 0.5,
                              ...) {
  if (object$input_skip == FALSE) (stop("Summary only applies to objects with input-skip = TRUE"))

  p <- object$sizes[1] # number of inputs
  L <- length(object$sizes) - 1 # number of layers (weight matrices incl. output)

  if (!is.numeric(epsilon) || length(epsilon) != 1 || epsilon <= 0) {
    stop("epsilon must be a positive numeric scalar")
  }
  if (!is.numeric(threshold) || length(threshold) != 1 || threshold <= 0 || threshold >= 1) {
    stop("threshold must be a numeric scalar in (0,1)")
  }

  active_counts <- matrix(nrow = p, ncol = L) # L0..L_{L-1}
  active_props <- matrix(nrow = p, ncol = L) # a0..a_{L-1}
  col_names_L <- c()
  col_names_a <- c()

  i <- 1
  for (l in object$layers$children) {
    W <- l$weight_mean$clone()$detach()$cpu()
    in_features <- dim(W)[2]
    cov_cols <- if (in_features == p) {
      1:p
    } else {
      (in_features - p + 1):in_features
    }
    W_cov <- W[, cov_cols] # out_features x p
    W_sq <- W_cov^2
    phi <- W_sq / (W_sq + epsilon^2)
    phi_mat <- as.matrix(phi)
    active_mat <- (phi_mat > threshold) # out_features x p (logical)
    counts <- colSums(active_mat)
    props <- counts / nrow(active_mat)

    active_counts[, i] <- counts
    active_props[, i] <- props
    col_names_L <- c(col_names_L, paste0("L", i - 1))
    col_names_a <- c(col_names_a, paste0("a", i - 1))
    i <- i + 1
  }

  # output layer
  W <- object$out_layer$weight_mean$clone()$detach()$cpu()
  in_features <- dim(W)[2]
  cov_cols <- if (in_features == p) {
    1:p
  } else {
    (in_features - p + 1):in_features
  }
  W_cov <- W[, cov_cols]
  W_sq <- W_cov^2
  phi <- W_sq / (W_sq + epsilon^2)
  phi_mat <- as.matrix(phi)
  active_mat <- (phi_mat > threshold)
  counts <- colSums(active_mat)
  props <- counts / nrow(active_mat)

  active_counts[, i] <- counts
  active_props[, i] <- props
  col_names_L <- c(col_names_L, paste0("L", i - 1))
  col_names_a <- c(col_names_a, paste0("a", i - 1))

  a_avg <- rowMeans(active_props)

  active_props <- round(active_props, 3)
  a_avg <- round(a_avg, 3)

  colnames(active_counts) <- col_names_L
  colnames(active_props) <- col_names_a

  summary_out <- as.data.frame(cbind(active_counts, active_props, a_avg))
  colnames(summary_out)[(ncol(active_counts) + ncol(active_props) + 1)] <- "a_avg"

  cat("Summary of SICNN_Net object (SIC-based):\n")
  cat("-----------------------------------\n")
  cat("L{l}: number of active weights connected to input xj in layer l\n")
  cat("-----------------------------------\n")
  cat("a{l}: proportion of those weights that are active\n")
  cat("-----------------------------------\n")
  cat("a_avg: average of a{l} across layers\n")
  cat("-----------------------------------\n")
  print(summary_out)
  cat(paste("The model took", object$elapsed_time, "seconds to train, using", object$device))

  # Global SIC weight count (active on active paths)
  glob_counts <- object$sic_weight_counts(epsilon = epsilon, threshold = threshold, active_paths = TRUE)
  cat(sprintf(
    "\nGlobal SIC Sparsity (Active Paths): %.2f%% (%d / %d active weights)\n",
    (glob_counts["removed"] / glob_counts["total"]) * 100,
    as.integer(glob_counts["active"]), as.integer(glob_counts["total"])
  ))
  invisible(summary_out)
}


#' @title Residuals from SICNN fit
#' @description Residuals from an object of the \code{SICNN_Net} class.
#' @param object An object of class \code{SICNN_Net}.
#' @param type Currently only 'response' is implemented i.e. y_true - y_predicted.
#' @param ... further arguments passed to or from other methods.
#' @return A numeric vector of residuals (\code{y_true - y_predicted})
#' @examples
#' \donttest{
#' x <- torch::torch_randn(3, 2)
#' b <- torch::torch_rand(2)
#' y <- torch::torch_matmul(x, b)
#' train_data <- torch::tensor_dataset(x, y)
#' train_loader <- torch::dataloader(train_data, batch_size = 3, shuffle = FALSE)
#' problem <- "regression"
#' sizes <- c(2, 1, 1)
#' model <- SICNN_Net(problem, sizes, input_skip = TRUE)
#' train_SICNN(epochs = 1, SICNN = model, lr = 0.01, train_dl = train_loader, n_train = 3)
#' residuals(model)
#' }
#' @export
residuals.SICNN_Net <- function(object, type = c("response"), ...) {
  y_true <- object$y
  y_predicted <- object$r
  if (type == "response") {
    return(y_true - y_predicted)
  } else {
    (stop("only y - y_pred residuals are currently implemented"))
  }
}


#' @title Get model coefficients (local explanations) of an \code{SICNN_Net} object
#' @description Given an input sample x_1,... x_j (with j the number of variables), the local explanation is found by
#' considering active paths. If relu activation functions are assumed, each path is a piecewise
#' linear function, so the contribution for x_j is just the sum of the weights associated with the paths connecting x_j to the output.
#' The contributions are found by taking the gradient wrt x.
#' @param object an object of class \code{SICNN_Net}.
#' @param dataset Either a \code{torch::dataloader} object, or a \code{torch::torch_tensor} object.
#' The former is assumed to be the same \code{torch::dataloader} used for training or testing.
#' The latter can be any user-defined data.
#' @param inds Optional integer vector of row indices in the dataset to compute explanations for.
#' @param output_neuron integer, which output neuron to explain (default = 1).
#' @param num_data integer, if no indices are chosen, the first \code{num_data} of \code{dataset} are automatically used for explanations.
#' @param num_data integer, if no indices are chosen, the first \code{num_data} of \code{dataset} are automatically used for explanations.
#' @param uncertainty logical, whether to compute uncertainty using the Delta method.
#' @param fisher_dataloader A \code{torch::dataloader} to compute the Fisher Information.
#' @param ... further arguments passed to or from other methods.
#' @details
#' \itemize{
#'   \item If \code{uncertainty = TRUE}, confidence intervals are computed using the Delta method (Frequentist).
#'   \item If \code{num_data > 1} and \code{uncertainty = TRUE}, confidence intervals incorporate both sampling variation across the dataset and estimation uncertainty (via the Delta method per sample).
#'   \item If \code{num_data > 1} and \code{uncertainty = FALSE}, confidence intervals are computed purely based on the empirical distribution of explanations across samples.
#'   \item The output is a data frame with row names as input variables
#'         (\code{x0}, \code{x1}, \code{x2}, ...) and columns giving mean and 95% confidence intervals for each variable.
#' }
#' @return A data frame with rows corresponding to input variables and the following columns:
#' \itemize{
#'   \item \code{lower}: lower bound of the 95% confidence interval
#'   \item \code{mean}: mean contribution of the variable
#'   \item \code{upper}: upper bound of the 95% confidence interval
#' }
#'
#' @examples
#' \donttest{
#' x <- torch::torch_randn(3, 2)
#' b <- torch::torch_rand(2)
#' y <- torch::torch_matmul(x, b)
#' train_data <- torch::tensor_dataset(x, y)
#' train_loader <- torch::dataloader(train_data, batch_size = 3, shuffle = FALSE)
#' problem <- "regression"
#' sizes <- c(2, 1, 1)
#' model <- SICNN_Net(problem, sizes, input_skip = TRUE)
#' train_SICNN(epochs = 1, SICNN = model, lr = 0.01, train_dl = train_loader, n_train = 3)
#' coef(model, dataset = x, num_data = 1)
#' }
#' @export
coef.SICNN_Net <- function(object, dataset, inds = NULL, output_neuron = 1, num_data = 1,
                           uncertainty = FALSE, fisher_dataloader = NULL, 
                           covariance_type = c("diagonal", "block-diagonal", "KFAC"),
                           use_pseudo_inverse = FALSE, ...) {
  if (output_neuron > object$sizes[length(object$sizes)]) stop(paste("output_neuron =", output_neuron, "can not be greater than", object$sizes[length(object$sizes)]))
  if (is.null(inds)) {
    all_means <- matrix(nrow = object$sizes[1], ncol = num_data)
  } else {
    all_means <- matrix(nrow = object$sizes[1], ncol = length(inds))
  }


  row_names <- c()
  for (i in 1:object$sizes[1]) {
    row_names <- c(row_names, paste("x", i - 1, sep = ""))
  }


  if (class(dataset)[1] == "dataloader") {
    X <- dataset$dataset$tensors[[1]]$clone()$detach()$cpu()
  } else if (class(dataset)[1] == "torch_tensor") {
    X <- dataset # should be a tensor with shape (num_data,p), but need to make sure it accepts MNIST or other img data
    if (length(dim(X)) == 1) {
      X <- X$unsqueeze(dim = 1)
    } # reshape (p) to shape (1,p)
    if (dim(X)[length(dim(X))] != object$sizes[1]) stop("the last index must have shape equal to p")
  } else {
    stop("dataset must be either a torch_tensor or a dataloader object")
  }

  if (is.null(inds)) {
    if (dim(X)[1] < num_data) stop(paste("num_data =", num_data, "can not be greater than the number of total data points,", dim(X)[1]))
    X_explain <- X[1:num_data, ]
  } else {
    inds <- as.integer(inds) # in case user sends a numeric vector
    inds <- unique(inds) # remove any duplicates
    if (dim(X)[1] < length(inds)) stop(paste("number of indecies =", length(inds), "can not be greater than the number of total data points,", dim(X)[1]))
    if (dim(X)[1] < max(inds)) stop(paste("the largest index =", max(inds), "can not be greater than the number of total data points,", dim(X)[1]))

    num_data <- length(inds)
    X_explain <- X[inds, ]
  }


  # Precompute Fisher if needed
  fisher <- NULL
  if (uncertainty && !is.null(fisher_dataloader)) {
    covariance_type <- match.arg(covariance_type)
    fisher <- get_fisher_information(object, fisher_dataloader, type = covariance_type, device = object$device)
  }

  if (num_data == 1) { # Point estimate + optional Delta method SE
    expl <- get_local_explanations_gradient(object, X_explain,
      uncertainty = uncertainty, fisher_dataloader = fisher_dataloader, 
      fisher = fisher, covariance_type = covariance_type, 
      use_pseudo_inverse = use_pseudo_inverse, device = object$device, ...
    )
    e <- as.numeric(expl$explanations[1, , output_neuron])

    if (uncertainty && !is.null(expl$se)) {
      se <- as.numeric(expl$se[, output_neuron])
      qs <- cbind(e - 1.96 * se, e, e + 1.96 * se)
    } else {
      qs <- cbind(e, e, e)
    }
    colnames(qs) <- c("lower", "mean", "upper")
    rownames(qs) <- row_names
    return(as.data.frame(qs))
  }

  # If num_data > 1
  all_means <- matrix(nrow = object$sizes[1], ncol = num_data)
  all_ses <- if (uncertainty) matrix(nrow = object$sizes[1], ncol = num_data) else NULL

  for (i in 1:num_data) {
    data <- X_explain[i, ]
    expl <- get_local_explanations_gradient(object, data,
      uncertainty = uncertainty, fisher_dataloader = fisher_dataloader,
      fisher = fisher, covariance_type = covariance_type, 
      use_pseudo_inverse = use_pseudo_inverse, device = object$device, ...
    )
    e <- expl$explanations
    all_means[, i] <- as.numeric(e[1, , output_neuron])
    if (uncertainty && !is.null(expl$se)) {
      all_ses[, i] <- as.numeric(expl$se[, output_neuron])
    }
  }

  rownames(all_means) <- row_names
  
  if (uncertainty && !is.null(all_ses)) {
    # Combine sampling variation and estimation uncertainty via resampling
    # For each variable, pool pseudo-samples from each data point's normal distribution
    n_resamples <- 1000 # samples per data point
    qs <- t(apply(matrix(1:object$sizes[1]), 1, function(j) {
      means <- all_means[j, ]
      ses <- all_ses[j, ]
      
      # Generate pool of pseudo-samples
      # For each data point i, draw samples from N(mean_i, se_i^2)
      pool <- unlist(lapply(seq_along(means), function(k) {
        stats::rnorm(n_resamples, mean = means[k], sd = ses[k])
      }))
      
      return(c(
        lower = as.numeric(stats::quantile(pool, 0.025)),
        mean = mean(pool),
        upper = as.numeric(stats::quantile(pool, 0.975))
      ))
    }))
  } else {
    qs <- t(apply(all_means, 1, quants))
  }
  
  colnames(qs) <- c("lower", "mean", "upper")
  rownames(qs) <- row_names

  return(as.data.frame(qs))
}


#' @title Obtain predictions from the variational posterior of an \code{SICNN model}
#' @description Draw from the (variational) posterior distribution of a trained \code{SICNN_Net} object.
#' @param object A trained \code{SICNN_Net} object
#' @param mpm logical, whether to use the median probability model.
#' @param newdata A \code{torch::dataloader} object containing the data with which to predict.
#' @param draws integer, the number of samples to draw from the posterior.
#' @param device character, device for computation (default = \code{"cpu"}).
#' @param link Optional link function to apply to the network output. Currently not implemented.
#' @param ... further arguments passed to or from other methods.
#' @return A \code{torch::torch_tensor}  of shape \code{(draws,N,C)} where \code{N} is the number of samples in \code{newdata}, and \code{C} the number of outputs.
#' @examples
#' \donttest{
#' x <- torch::torch_randn(3, 2)
#' b <- torch::torch_rand(2)
#' y <- torch::torch_matmul(x, b)
#' train_data <- torch::tensor_dataset(x, y)
#' train_loader <- torch::dataloader(train_data, batch_size = 3, shuffle = FALSE)
#' problem <- "regression"
#' sizes <- c(2, 1, 1)
#' model <- SICNN_Net(problem, sizes, input_skip = TRUE)
#' train_SICNN(epochs = 1, SICNN = model, lr = 0.01, train_dl = train_loader, n_train = 3)
#' predict(model, mpm = FALSE, newdata = train_loader, draws = 1)
#' }
#' @export
predict.SICNN_Net <- function(object, newdata, mpm = FALSE, draws = 10, device = "cpu", link = NULL, ...) { # should newdata be a dataloader or a dataset?
  object$eval()
  object$raw_output <- TRUE # skip final sigmoid/softmax
  if (!object$computed_paths) {
    if (object$input_skip) {
      if (!is.null(object$criterion_trained) && object$criterion_trained == "SIC") {
        object$compute_paths_input_skip(epsilon = object$sic_epsilon_T, threshold = object$sic_threshold)
      } else {
        object$compute_paths_input_skip()
      }
    } else {
      if (!is.null(object$criterion_trained) && object$criterion_trained == "SIC") {
        object$compute_paths(epsilon = object$sic_epsilon_T, threshold = object$sic_threshold)
      } else {
        object$compute_paths()
      }
    }
  }
  if (class(newdata)[[1]] != "dataloader") stop("Currently only torch::dataloader objects are supported for newdata")
  out_shape <- object$sizes[length(object$sizes)] # number of output neurons
  all_outs <- NULL
  torch::with_no_grad({
    coro::loop(for (b in newdata) {
      outputs <- torch::torch_zeros(draws, dim(b[[1]])[1], out_shape)$to(device = device)
      for (i in 1:draws) {
        data <- b[[1]]$to(device = device)
        outputs[i] <- object(data, sparse = mpm)
      }
      all_outs <- torch::torch_cat(c(all_outs, outputs), dim = 2) # add all the mini-batches together
    })
  })
  return(all_outs)
}


#' @title Print summary of an \code{SICNN_Net} object
#' @description
#' Provides a summary of a trained \code{SICNN_Net} object.
#' Includes the model type (input-skip or not), whether normalizing flows
#' are used, module and sub-module structure, number of trainable parameters, and prior
#' variance and inclusion probabilities for the weights.
#' @param x An object of class \code{SICNN_Net}.
#' @param ... Further arguments passed to or from other methods.
#' @return Invisibly returns the input \code{x}.
#' @examples
#' \donttest{
#' x <- torch::torch_randn(3, 2)
#' b <- torch::torch_rand(2)
#' y <- torch::torch_matmul(x, b)
#' train_data <- torch::tensor_dataset(x, y)
#' train_loader <- torch::dataloader(train_data, batch_size = 3, shuffle = FALSE)
#' problem <- "regression"
#' sizes <- c(2, 1, 1)
#' model <- SICNN_Net(problem, sizes, input_skip = TRUE)
#' print(model)
#' }
#' @export
print.SICNN_Net <- function(x, ...) {
  module_info <- x$modules[[1]]

  # Model description
  model_name <- if (isTRUE(x$input_skip)) {
    "SICNN with input-skip"
  } else {
    "SICNN without input-skip"
  }

  flow <- if (isTRUE(x$flow)) {
    "with normalizing flows"
  } else {
    "without normalizing flows"
  }

  # Header
  cat("\n========================================\n")
  cat("          SICNN Model Summary           \n")
  cat("========================================\n\n")

  # Module info
  total_params <- sum(sapply(module_info$parameters, length))
  cat("Module Overview:\n")
  cat("  - An `nn_module` containing", total_params, "parameters.\n\n")

  # Submodules
  cat("---------------- Submodules ----------------\n")
  submodules <- module_info$modules
  if (length(submodules) == 0) {
    cat("  No submodules detected.\n")
  } else {
    for (name in names(submodules)) {
      mod <- submodules[[name]]
      if (is.null(mod)) next
      n_params <- if (!is.null(mod$parameters)) sum(sapply(mod$parameters, length)) else 0
      cat(sprintf("  - %-20s : %-15s # %d parameters\n", name, class(mod)[1], n_params))
    }
  }

  # Model details
  cat("\nModel Configuration:\n")
  cat("  -", model_name, "\n")
  cat("  - Optimized using variational inference", flow, "\n\n")

  # Priors
  cat("Priors:\n")
  cat(
    "  - Prior inclusion probabilities per layer: ",
    paste(x$prior_inclusion, collapse = ", "), "\n"
  )
  cat(
    "  - Prior std dev for weights per layer:    ",
    paste(x$prior_std, collapse = ", "), "\n"
  )

  cat("\n=================================================================\n\n")
  invisible(x)
}


#' @title Plot \code{SICNN_Net} objects
#' @description
#' Given a trained \code{SICNN_Net} model, this function produces either:
#' \itemize{
#'   \item \strong{Global plot}: a visualization of the network structure,
#'     showing only the active paths.
#'   \item \strong{Local explanation}: a plot of the local
#'     explanation for a single input sample, optionally including
#'     error bars from the Delta method.
#' }
#' @param x An instance of \code{SICNN_Net}.
#' @param type Either \code{"global"} or \code{"local"}.
#' @param data If local is chosen, one sample must be provided to obtain the explanation. Must be a \code{torch::torch_tensor} of shape \code{(1,p)}.
#' @param data If local is chosen, one sample must be provided to obtain the explanation. Must be a \code{torch::torch_tensor} of shape \code{(1,p)}.
#' @param uncertainty logical, whether to include Delta method uncertainty.
#' @param fisher_dataloader A \code{torch::dataloader} to compute the Fisher Information.
#' @param covariance_type character, the type of Fisher Information approximation: \code{"diagonal"}, \code{"block-diagonal"}, or \code{"KFAC"}.
#' @param use_pseudo_inverse logical, whether to use the pseudo-inverse if the Fisher Information is singular.
#' @param ... further arguments passed to or from other methods.
#' @return No return value. Called for its side effects of producing a plot.
#' @export
plot.SICNN_Net <- function(x, type = c("global", "local"), data = NULL, ...) {
  if (x$input_skip == FALSE) (stop("Plotting currently only implemented for input-skip"))
  if (x$computed_paths == FALSE) {
    if (x$input_skip) {
      if (!is.null(x$criterion_trained) && x$criterion_trained == "SIC") {
        x$compute_paths_input_skip(epsilon = x$sic_epsilon_T, threshold = x$sic_threshold)
      } else {
        x$compute_paths_input_skip()
      }
    } else {
      if (!is.null(x$criterion_trained) && x$criterion_trained == "SIC") {
        x$compute_paths(epsilon = x$sic_epsilon_T, threshold = x$sic_threshold)
      } else {
        x$compute_paths()
      }
    }
  }
  d <- match.arg(type)
  if (d == "global") {
    SICNN_plot(x, ...)
  } else {
    if (is.null(data)) stop("data must contain a sample to explain")
    plot_local_explanations_gradient(x, input_data = data, device = x$device, ...)
  }
}
