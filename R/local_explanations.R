library(torch)
library(ggplot2)
library(svglite)
library(Matrix)

#' @title Function to get gradient based local explanations for input-skip SICNNs.
#' @description Works by computing the gradient wrt to input, given we have
#' relu activation functions.
#' @param model A \code{SICNN_Net} with input-skip
#' @param input_data The data to be explained (one sample).
#' @param magnitude If TRUE, only return explanations. If FALSE, multiply by input values.
#' @param include_potential_contribution IF TRUE, If covariate=0, 
#' we assume that the contribution is negative (good/bad that it is not included)
#' if FALSE, just removes zero covariates.
#' @param device character, the device to be trained on. Default is 'cpu', can be 'mps' or 'gpu'. 
#' @param uncertainty logical, whether to compute uncertainty using the Delta method.
#' @param fisher_dataloader A \code{torch::dataloader} to compute the Fisher Information. If NULL and uncertainty is TRUE, it defaults to training data if available.
#' @param covariance_type character, the type of Fisher Information approximation: \code{"diagonal"}, \code{"block-diagonal"}, or \code{"KFAC"}.
#' @param use_pseudo_inverse logical, whether to use the pseudo-inverse if the Fisher Information is singular.
#' @param ... further arguments (e.g., deprecated num_samples periodically used in old scripts).
#' @return A list with the following elements:
#'   \describe{
#'     \item{explanations}{A \code{torch::tensor} of shape (1, p, num_classes).}
#'     \item{p}{integer, the number of input features.}
#'     \item{predictions}{A \code{torch::tensor} of shape (1, num_classes).}
#'     \item{se}{A \code{torch::tensor} of shape (p, num_classes) containing standard errors (if uncertainty=TRUE).}
#'   }
#' @export
get_local_explanations_gradient <- function(model,input_data, 
                                            magnitude = TRUE,
                                            include_potential_contribution = FALSE,
                                            device = 'cpu',
                                            uncertainty = FALSE,
                                            fisher_dataloader = NULL,
                                            covariance_type = c("diagonal", "block-diagonal", "KFAC"),
                                            use_pseudo_inverse = FALSE, 
                                            fisher = NULL, ...){
  covariance_type <- match.arg(covariance_type)
  if(model$input_skip == FALSE)(stop('This function is only implemented for input-skip'))
  if(model$computed_paths == FALSE){
    model$compute_paths_input_skip() #need this before computing explanations
  }
  #need to make sure input_data comes in shape (1,p) where p is #input variables
  num_classes <- model$sizes[length(model$sizes)]
  if(input_data$dim() == 4){ #in the case of MNIST or other image data
    (input_data <- input_data$view(c(-1,dim(input_data)[3]*dim(input_data)[4]))) #reshape to(1,p)
    
  }
  else{ #in case shape is either (p) or (1,p), both cases returns (1,p)
    input_data <- input_data$view((c(-1,length(input_data))))
  }
  
  p <- input_data$shape[2] #number of variables
  explanations <- torch::torch_zeros(1,p,num_classes)
  predictions <- torch::torch_zeros(1,num_classes)
  
  model$raw_output = TRUE #to skip last sigmoid/softmax layer
  
  # Point estimate explanation (one pass)
  input_data$requires_grad = TRUE
  input_data <- input_data$to(device = device)
  model$zero_grad()
  output = model(input_data, sparse = TRUE) #forward pass, using sparse
  for(k in 1:num_classes){
    output_value <- output[1,k]
    grad = torch::autograd_grad(outputs = output_value,inputs = input_data,
                                grad_outputs = torch::torch_ones_like(output_value),
                                retain_graph =TRUE)
    
    explanations[1,,k] <- grad[[1]]
    predictions[1,k] <- output[1,k]
  }

  inds <- torch::torch_nonzero(input_data == 0)[,2] #find index of 0s in input data
  if(include_potential_contribution){
      explanations[,inds] <- - explanations[,inds]
  }
  else{#remove variables that do not contribute to predictions
    explanations[,inds] <- 0
  }
  if(! magnitude){
    explanations <- explanations * input_data$view(c(1,-1,1))
  }
  predictions <- predictions$detach()$cpu()
  outputs = list('explanations' = explanations,'p' = p,'predictions' = predictions)
  model$raw_output <- FALSE  # restore after explanation pass

  if (uncertainty) {
    if (is.null(fisher_dataloader)) {
      stop("fisher_dataloader must be provided if uncertainty = TRUE")
    }
    
    # 1. Compute Fisher Information (Sigma_w approximation)
    if (is.null(fisher)) {
        fisher <- get_fisher_information(model, fisher_dataloader, type = covariance_type, device = device)
    }
    
    # 2. Compute Jacobian of g(w,x) = d mu(w,x)/dx wrt w using autograd
    # Use raw linear predictor (before sigmoid/softmax), consistent with explanations
    model$raw_output <- TRUE
    
    # Collect all leaf parameter tensors once (requires_grad = TRUE already)
    vars <- list()
    for (v in model$parameters) {
      vars <- c(vars, list(v))
    }
    
    se_matrix <- torch::torch_zeros(p, num_classes, device = device)
    
    for (k in 1:num_classes) {
      model$zero_grad()
      # Fresh forward pass with full autograd graph for this class
      input_data_tensor <- input_data$view(c(1, -1))$to(device = device)
      input_data_tensor$requires_grad <- TRUE
      output_k_full <- model(input_data_tensor, sparse = TRUE)
      output_k <- output_k_full[1, k]
      
      # First-order gradient: d mu_k / dx  (shape 1 x p)
      # create_graph=TRUE keeps the graph so we can differentiate again wrt weights
      grad_x <- torch::autograd_grad(output_k, input_data_tensor, create_graph = TRUE)[[1]]
      
      # Var(g_j(w)) = J_gj * Sigma_w * J_gj^T
      # J_gj = d(grad_x[j])/dw computed via backward() per feature j.
      # We use backward() + param$grad instead of autograd_grad(allow_unused=TRUE)
      # because allow_unused=TRUE returns undefined C++ tensors that look non-NULL
      # to R but cause "Expected a proper Tensor" errors when used in operations.
      var_g_k <- torch::torch_zeros(p, device = device)
      
      for (j in 1:p) {
        # Zero parameter gradients before each backward pass
        model$zero_grad()
        # backward from scalar grad_x[1,j] populates param$grad with d(grad_x[j])/dw
        # retain_graph=TRUE to keep graph alive for next j
        grad_x[1, j]$backward(retain_graph = TRUE)
        
        if (covariance_type == "diagonal") {
          current_var <- torch::torch_tensor(0, device = device)
          for (l in seq_along(vars)) {
            g_l <- vars[[l]]$grad  # proper R NULL for genuinely unused params
            if (is.null(g_l)) next
            # f_diag now contains the sandwich covariance V_j
            v_diag <- fisher$diag[[l]]
            # Var(beta_hat)_j = V_j; zero out pruned weights (post-selection inference)
            # Use small threshold on precision or variance? 
            # In SIC, pruned weights have very small variance.
            # We keep the threshold logic if needed, but the sandwich naturally handles it.
            current_var <- current_var + torch::torch_sum(g_l^2 * v_diag)
          }
          var_g_k[j] <- current_var$detach()
          
        } else if (covariance_type == "block-diagonal") {
          current_var <- torch::torch_tensor(0, device = device)
          for (l in seq_along(vars)) {
            g_l <- vars[[l]]$grad
            if (is.null(g_l)) next
            # f_block now contains the sandwich covariance block V_block
            sigma_w_block <- fisher$blocks[[l]]
            
            g_vec <- g_l$view(c(1L, -1L))
            current_var <- current_var +
              torch::torch_matmul(
                torch::torch_matmul(g_vec, sigma_w_block),
                torch::torch_t(g_vec)
              )
          }
          var_g_k[j] <- current_var$detach()
          
        } else if (covariance_type == "KFAC") {
          stop("KFAC uncertainty not yet fully implemented, please use 'diagonal' or 'block-diagonal'")
        }
      }
      se_matrix[, k] <- torch::torch_sqrt(torch::torch_clamp(var_g_k, min = 0))$cpu()
    }
    model$raw_output <- FALSE  # restore after uncertainty pass
    model$zero_grad()           # clean up gradients
    outputs$se <- se_matrix
  }

  return(outputs)
}


#' @title Compute parameter covariance for an SICNN model.
#' @description Computes the covariance matrix (or its approximation) for the weights of a trained SICNN model using the sandwich estimator (Fan and Li, 2001) for penalized models.
#' @param model A trained \code{SICNN_Net} object.
#' @param dataloader A \code{torch::dataloader} object.
#' @param type character, the type of approximation: \code{"diagonal"} or \code{"block-diagonal"}.
#' @param device character, the device to be used (default = \code{"cpu"}).
#' @return A list containing the covariance matrix components (diagonal or blocks).
#' @export
get_fisher_information <- function(model, dataloader, type = c("diagonal", "block-diagonal", "KFAC"), device = "cpu") {
  type <- match.arg(type)
  # Ensure raw_output is FALSE so we use the full log-likelihood for Fisher
  model$raw_output <- FALSE
  model$eval()
  model$to(device = device)
  
  params <- list()
  for (p in model$parameters) {
    params <- c(params, list(p))
  }
  
  # Accumulate I0 (Empirical Fisher of data part)
  # Initialize both components to avoid NULL access later
  fisher <- list()
  fisher$diag <- lapply(params, function(p) torch::torch_zeros_like(p))
  if (type == "block-diagonal") {
    fisher$blocks <- lapply(params, function(p) {
      n <- p$numel()
      torch::torch_zeros(n, n, device = device)
    })
  }

  is_sic <- (!is.null(model$criterion_trained) && model$criterion_trained == "SIC")
  
  # SIC-specific parameters
  # Fall back to 1 if n_train is missing (e.g. model not trained with SIC or older version)
  n_train_ref <- if (is_sic && !is.null(model$n_train)) model$n_train else 1
  sic_penalty <- if (is_sic && !is.null(model$sic_penalty)) model$sic_penalty else 0
  sic_eps <- if (is_sic && !is.null(model$sic_epsilon_T)) model$sic_epsilon_T else 1e-5

  n_total <- 0
  mse_total <- 1
  if (is_sic) {
    # Efficient pass for counts and mean-squared error if needed
    total_loss <- 0
    torch::with_no_grad({
      coro::loop(for (b in dataloader) {
        batch_n <- dim(b[[1]])[1]
        n_total <- n_total + batch_n
        if (model$problem_type == "regression") {
            data_b <- b[[1]]$to(device = device)
            target_b <- b[[2]]$to(device = device)
            output_b <- model(data_b, sparse = TRUE)
            total_loss <- total_loss + model$loss_fn(output_b$squeeze(), target_b$to(torch::torch_float()))$item()
        }
      })
    })
    if (model$problem_type == "regression") {
        mse_total <- if (n_total > 0) total_loss / n_total else 1e-12
    }
  } else {
    # Minimal pass for baseline n_total
    coro::loop(for (b in dataloader) {
      n_total <- n_total + dim(b[[1]])[1]
    })
  }
  
  # 1. Main pass: compute per-sample gradients and pool Information
  coro::loop(for (b in dataloader) {
    data <- b[[1]]$to(device = device)
    target <- b[[2]]$to(device = device)
    batch_size <- dim(data)[1]

    # For per-sample gradients, we loop through the batch
    for (i in 1:batch_size) {
      model$zero_grad()
      input_i  <- data[i]$unsqueeze(dim = 1)  # shape (1, p)
      target_i <- target[i]$unsqueeze(dim = 1)
      
      output_i <- model(input_i, sparse = TRUE)
      
      if (model$problem_type == 'multiclass classification' | model$problem_type == 'MNIST') {
         # Target needs to be a long tensor for NLL loss
         loss <- model$loss_fn(output_i, target_i$to(torch::torch_long()))
      } else if (model$problem_type == 'binary classification') {
         loss <- model$loss_fn(output_i$squeeze(), target_i$to(torch::torch_float())$view(output_i$squeeze()$shape))
      } else { # regression
         loss <- model$loss_fn(output_i$squeeze(), target_i$to(torch::torch_float())$view(output_i$squeeze()$shape))
      }
      
      loss$backward()
      
      # Scale gradient according to SIC objective
      # SIC_loss = Scaled_Data_Loss + Penalty
      # Gradient contribution from data sample i:
      grad_scale <- 1
      if (is_sic) {
        if (model$problem_type == "regression") {
            # Total Data Loss = n_train * log(RSS/n_total)
            # Grad = n_train * (1/RSS) * sum(grad_i) = (n_train / (n_total * MSE)) * sum(grad_i)
            grad_scale <- n_train_ref / (n_total * mse_total)
        } else {
            # Total Data Loss = (n_train / n_total) * sum(NLL_i)
            grad_scale <- n_train_ref / n_total
        }
      }

      if (type == "diagonal") {
        for (j in seq_along(params)) {
          g <- params[[j]]$grad
          if (!is.null(g)) {
            fisher$diag[[j]] <- fisher$diag[[j]] + (g * grad_scale)^2
          }
        }
      } else if (type == "block-diagonal") {
        for (j in seq_along(params)) {
          g <- params[[j]]$grad
          if (!is.null(g)) {
            g_vec <- (g * grad_scale)$view(-1)
            fisher$blocks[[j]] <- fisher$blocks[[j]] + torch::torch_outer(g_vec, g_vec)
          }
        }
      }
    }
  })
  
  # 2. Compute Sandwich Covariance V = H^-1 I0 H^-1
  # Where H = I0 + Hp
  for (j in seq_along(params)) {
    I0_diag <- fisher$diag[[j]]
    
    # Calculate penalty Hessian contribution
    if (is_sic && sic_penalty > 0) {
      w <- params[[j]]
      # k''(w) = 2*eps^2 * (eps^2 - 3*w^2) / (w^2 + eps^2)^3
      w2 <- w^2
      eps2 <- sic_eps^2
      h_pen <- sic_penalty * 2 * eps2 * (eps2 - 3 * w2) / (w2 + eps2)^3
      h_pen <- torch::torch_clamp(h_pen, min = 0) # ensure positive definite penalty curvature
      
      H_diag <- I0_diag + h_pen
      # Sandwich diagonal: V_j = I0_j / (H_j)^2
      fisher$diag[[j]] <- I0_diag / (H_diag^2 + 1e-12)
      
      if (type == "block-diagonal") {
        I0_block <- fisher$blocks[[j]]
        n_block <- I0_block$shape[1]
        H_block <- I0_block + torch::torch_diag(h_pen$view(-1))
        # solve(H) %*% I0 %*% solve(H)
        H_inv <- torch::torch_inverse(H_block + torch::torch_eye(n_block, device = device) * 1e-6)
        fisher$blocks[[j]] <- torch::torch_matmul(torch::torch_matmul(H_inv, I0_block), H_inv)
      }
    } else {
      # Baseline (unpenalized): V = I0^-1
      fisher$diag[[j]] <- 1 / (I0_diag + 1e-12)
      if (type == "block-diagonal") {
        I0_block <- fisher$blocks[[j]]
        n_block <- I0_block$shape[1]
        fisher$blocks[[j]] <- torch::torch_inverse(I0_block + torch::torch_eye(n_block, device = device) * 1e-6)
      }
    }
  }

  return(fisher)
}





#' @title Function to obtain empirical 95% confidence interval, including the mean
#' @description Using the built in quantile function to return 95% confidence interval
#' @param x numeric vector whose sample quantiles is desired.
#' @return The quantiles in addition to the mean.
#' @export
quants <- function(x){ #maybe should allow for something other than 95% CI
  out <- c(
    lower = stats::quantile(x, 0.025),
    mean  = mean(x),
    upper = stats::quantile(x, 0.975)
  )
  return(out) 
}


#' @title Plot the gradient based local explanations for one sample with input-skip SICNNs.
#' @description Plots the contribution of each covariate, and the prediction, with error bars. 
#' @param model An instance of \code{SICNN_Net} with input-skip enabled.
#' @param input_data The data to be explained (one sample).
#' @param input_data The data to be explained (one sample).
#' @param device character, the device to be trained on. Default is cpu. Can be 'mps' or 'gpu'.
#' @param uncertainty logical, whether to include Delta method uncertainty.
#' @param fisher_dataloader A \code{torch::dataloader} to compute the Fisher Information.
#' @param covariance_type character, the type of Fisher Information approximation: \code{"diagonal"}, \code{"block-diagonal"}, or \code{"KFAC"}.
#' @param use_pseudo_inverse logical, whether to use the pseudo-inverse if the Fisher Information is singular.
#' @param save_svg the path where the plot will be saved as svg, if save_svg is not NULL.
#' @param ... further arguments passed to or from other methods.
#' @return This function produces plots as a side effect and does not return a value.
#' @export
plot_local_explanations_gradient <- function(model,input_data,device = 'cpu',
                                             uncertainty = FALSE, fisher_dataloader = NULL,
                                             covariance_type = c("diagonal", "block-diagonal", "KFAC"),
                                             use_pseudo_inverse = FALSE,
                                             save_svg = NULL, ...){
  outputs <- get_local_explanations_gradient(model = model,input_data = input_data,
                                             device = device,
                                             uncertainty = uncertainty, 
                                             fisher_dataloader = fisher_dataloader,
                                             covariance_type = covariance_type,
                                             use_pseudo_inverse = use_pseudo_inverse, ...)
  
  preds<- as.matrix(outputs$predictions) #shape (1,num_classes)
  expl <- as.array(outputs$explanations) #shape (1,p,num_classes)


  
  num_classes <- model$sizes[length(model$sizes)]
  for(cls in seq_len(num_classes)){ #loop over each class and compute quantiles
    
    
    expl_class <- expl[1,,cls]
    
    pred_val <- preds[1, cls]
    
    names <- c()
    median <- expl_class
    pred_median <- pred_val
    
    if (uncertainty && !is.null(outputs$se)) {
      se_class <- as.numeric(outputs$se[, cls])
      # Delta method CI: mean +/- 1.96 * SE
      min <- c(median - 1.96 * se_class, pred_val) # prediction uncertainty not included in Delta yet?
      max <- c(median + 1.96 * se_class, pred_val)
    } else {
      min <- c(median, pred_val)
      max <- c(median, pred_val)
    }
    
    contribution <- c(median,pred_median)
    
    
    num_inputs <- model$sizes[1]
    for(x in seq_len(num_inputs)){#get names for x-axis
      name <- paste('x',x-1,sep = '')
      names <- c(names,name)
      
    }
    output <- 'out'
    names <- c(names,output)
  

    data <- data.frame(
      name=names,
      contribution = contribution,
      min = min,
      max = max
    )
   
   
    plt <- ggplot2::ggplot(data <- data,ggplot2::aes(x=factor(name,levels = name),
                            y=contribution,
                            fill=factor(ifelse(name=="out","out","input variables")))) +
      ggplot2::geom_bar(stat="identity") +
      ggplot2::scale_fill_manual(name = paste("Output neuron",cls), values=c("#D5E8D4",'#F8CECC')) +
      ggplot2::geom_errorbar(ggplot2::aes(x=name,ymin=min, ymax=max), width=0.6, colour="black", alpha=0.9, linewidth=0.5) +
      ggplot2::xlab("")+ggplot2::ylab('Contribution')+ggplot2::ggtitle('Local explanation, with 95% empirical confidence bars')
    print(plt)
    if(!is.null(save_svg)){
      new_file <- file.path(
        dirname(save_svg),
        paste0(sub("\\.[^.]*$", "", basename(save_svg)), "_class_", cls, ".svg") #check how this works
      )
      svglite::svglite(new_file, width=8, height=6)
      print(plt)
      grDevices::dev.off()
      message("SVG saved to: ",new_file)
    } 
    
    
    
    

    
  
   
    
    
  } 



}





