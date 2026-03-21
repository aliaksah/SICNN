library(torch)

#' @title Train an instance of \code{SICNN_Net}.
#' @description
#' Function that for each epoch iterates through each mini-batch, computing
#' the loss and using back-propagation to update the network parameters.
#'
#' By default, \code{train_SICNN} uses the original variational Bayes objective:
#' the data-fit loss plus the KL-divergence term from \code{SICNN$kl_div()}.
#' Alternatively, setting \code{criterion = "SIC"} replaces the KL term with the
#' smooth BIC-type penalty of O’Neill and Burke (2023), i.e.
#' \deqn{ -\ell(\theta) + \tfrac{1}{2}\log(n_\mathrm{train}) \|\theta\|_{0,\epsilon}, }
#' where \eqn{\|\theta\|_{0,\epsilon}} is the smooth L0 norm computed via
#' \code{SICNN$smooth_param_count(epsilon)} and \eqn{n_\mathrm{train}} is the number
#' of training observations. During training with \code{criterion = "SIC"}, an
#' \eqn{\epsilon}-telescope is implemented across epochs, as recommended in the paper.
#'
#' @param epochs integer, total number of epochs to train for, where one epoch is a pass through the entire training dataset (all mini batches).
#' @param SICNN An instance of  \code{SICNN_Net}, to be trained.
#' @param lr numeric, the learning rate to be used in the Adam optimizer.
#' @param train_dl An instance of \code{torch::dataloader} consisting of a tensor dataset
#' with features and targets.
#' @param device the device to be trained on. Default is 'cpu', also accepts 'gpu' or 'mps'.
#' @param scheduler A torch learning rate scheduler object. Can be used to decay learning rate for better convergence, 
#' currently only supports 'step'.
#' @param sch_step_size Where to decay if using \code{torch::lr_step}. E.g. 1000 means learning rate is decayed every 1000 epochs.
#' @param criterion character, either \code{"VI"} (default) for the original variational
#' Bayes objective with KL-divergence, or \code{"SIC"} for the smooth information
#' criterion of O’Neill and Burke (2023), which replaces the KL term with a smooth
#' BIC-type penalty based on the smooth L0 norm.
#' @param n_train integer, total number of training observations used when
#' \code{criterion = "SIC"} to scale the BIC penalty via \eqn{\log(n_\mathrm{train})/2}.
#' Ignored when \code{criterion = "VI"}.
#' @param epsilon_1 numeric, starting value of the \eqn{\epsilon}-telescope when using
#' \code{criterion = "SIC"}. Defaults to 10, as in O’Neill and Burke (2023).
#' @param epsilon_T numeric, final value of the \eqn{\epsilon}-telescope when using
#' \code{criterion = "SIC"}. Defaults to 1e-5.
#' @param steps_T integer, number of steps in the \eqn{\epsilon}-telescope sequence
#' when using \code{criterion = "SIC"}. Defaults to 100.
#' @param sic_threshold numeric scalar in (0,1) used for reporting “active”
#' edges under SIC (i.e., in thresholding \eqn{\phi_\epsilon(w_\mathrm{eff})}).
#' This does not change the objective; it only affects the reported density/sparsity.
#' @param sic_report_epsilon character, either \code{"current"} or \code{"final"}:
#'   controls whether training logs use the current \eqn{\epsilon} from the
#'   telescope, or always use \eqn{\epsilon_T} (the final level).
#' @return a list containing the losses and accuracy (if classification) and density for each epoch during training.
#' For comparisons sake we show the density with and without active paths.
#' @examples
#' \donttest{ 
#'x<-torch::torch_randn(3,2) 
#'b <- torch::torch_rand(2)
#'y <- torch::torch_matmul(x,b)
#'train_data <- torch::tensor_dataset(x,y)
#'train_loader <- torch::dataloader(train_data,batch_size = 3,shuffle=FALSE)
#'problem<-'regression'
#'sizes <- c(2,1,1) 
#'inclusion_priors <-c(0.9,0.2) 
#'inclusion_inits <- matrix(rep(c(-10,10),2),nrow = 2,ncol = 2)
#'stds <- c(1.0,1.0)
#'model <- SICNN_Net(problem,sizes,inclusion_priors,stds,inclusion_inits,flow = FALSE)
#'output <- train_SICNN(epochs = 1,SICNN = model, lr = 0.01,train_dl = train_loader)
#'}
#' @return A list with elements (returned invisibly):
#'   \describe{
#'     \item{accs}{Vector of accuracy per epoch (classification only).}
#'     \item{loss}{Vector of average loss per epoch.}
#'     \item{density}{Vector of network densities per epoch.}
#'   }
#'@export
train_SICNN <- function(epochs,
                        SICNN,
                        lr,
                        train_dl,
                        device = "cpu",
                        scheduler = NULL,
                        sch_step_size = NULL,
                        n_train = NULL,
                        epsilon_1 = 10,
                        epsilon_T = 1e-5,
                        steps_T = 100,
                        sic_threshold = 0.5,
                        sic_report_epsilon = c("final", "current"),
                        restarts = 1,
                        penalty = NULL){
  sic_report_epsilon <- match.arg(sic_report_epsilon)
  
  if (is.null(n_train) || !is.numeric(n_train) || length(n_train) != 1 || n_train <= 0) {
    stop("n_train must be a positive numeric scalar giving the number of training observations")
  }
  if (!is.numeric(epsilon_1) || !is.numeric(epsilon_T) || epsilon_1 <= 0 || epsilon_T <= 0) {
    stop("epsilon_1 and epsilon_T must be positive numerics")
  }
  if (!is.numeric(steps_T) || length(steps_T) != 1 || steps_T < 1) {
    stop("steps_T must be a positive integer")
  }
    # exponential epsilon-telescope as in O'Neill and Burke (2023)
    eps_seq <- epsilon_1 * (epsilon_T/epsilon_1) ^ ((0:(steps_T - 1)) / max(1, steps_T - 1))
    # penalty coefficient: default is log(n) (BIC-like), user can override for stronger/weaker sparsity
    if (is.null(penalty)) {
      sic_penalty <- log(n_train)
    } else {
      if (!is.numeric(penalty) || length(penalty) != 1 || penalty <= 0) {
        stop("penalty must be a positive numeric scalar")
      }
      sic_penalty <- penalty
    }
  SICNN$criterion_trained <- "SIC"
  SICNN$sic_epsilon_T <- epsilon_T
  SICNN$sic_epsilon_1 <- epsilon_1
  SICNN$sic_steps_T <- steps_T
  SICNN$sic_threshold <- sic_threshold
  SICNN$sic_report_epsilon <- sic_report_epsilon
  SICNN$sic_penalty <- sic_penalty
  opt <- torch::optim_adam(SICNN$parameters,lr = lr)
  if(! is.null(scheduler)){
    if(scheduler == 'step'){
      sl <- torch::lr_step(opt,step_size = sch_step_size,gamma = 0.1)
    }
  }
  
  SICNN$elapsed_time <- 0
  start <- base::proc.time()
  
  best_loss <- Inf
  best_state <- NULL
  best_l <- NULL
  
  for (r in 1:restarts) {
    if (restarts > 1) {
      cat(sprintf("\n--- Restart %d/%d ---\n", r, restarts))
      p_seq <- seq(0.01, 0.99, length.out = restarts)
      p <- p_seq[r]
      apply_sic_mask <- function(layer, prob) {
        layer$reset_parameters()
        mask <- (torch::torch_rand_like(layer$weight_mean) < prob)$to(torch::torch_float32())
        layer$weight_mean$data()$mul_(mask)
      }
      
      for (l in SICNN$layers$children) {
        apply_sic_mask(l, p)
      }
      apply_sic_mask(SICNN$out_layer, p)
      opt <- torch::optim_adam(SICNN$parameters, lr = lr)
      if(!is.null(scheduler)){
        if(scheduler == 'step'){
          sl <- torch::lr_step(opt,step_size = sch_step_size,gamma = 0.1)
        }
      }
    }
    
    accs <- c()
    losses <-c()
    density <- c()
    sparsity_pct <- c()
    active_weights <- c()
    removed_weights <- c()
    total_weights <- c()
    
    for (epoch in 1:epochs) {
  #  if(SICNN$input_skip){SICNN$compute_paths_input_skip()}
  #  else(SICNN$compute_paths)
    if(epoch == epochs){ #only need these at the last epoch for residuals
     SICNN$y <- c()
    SICNN$r <- c()}
 
    
    SICNN$train()
    corrects <- 0
    totals <- 0
    train_loss <- c()
    # map epoch to epsilon index
    idx <- min(steps_T, max(1, ceiling(epoch * steps_T/epochs)))
    epsilon <- eps_seq[idx]
    epsilon_report <- if (sic_report_epsilon == "final") epsilon_T else epsilon
    # use coro::loop() for stability and performance
    coro::loop(for (b in train_dl) {

      opt$zero_grad()
      data <- b[[1]]$to(device = device)
      output <- SICNN(data, sparse=FALSE)
      target <- b[[2]]$to(device=device)
      if(epoch == epochs){ #add the targets and outputs to y and r
        SICNN$y <- c(SICNN$y, as.numeric(target$clone()$detach()$cpu()))
        SICNN$r <- c(SICNN$r,as.numeric(output$clone()$detach()$squeeze()$cpu()))}
      
      
 
      if(SICNN$problem_type == 'multiclass classification'| SICNN$problem_type == 'MNIST'){ #nll loss needs float tensors but bce loss needs long tensors 
        target <- torch::torch_tensor(target,dtype = torch::torch_long())
      }
      else(output <- output$squeeze()) #remove last dimension from binary classifiction or regression
      data_loss <- SICNN$loss_fn(output, target)
      
      # ---- SIC-consistent loss scaling ----
      # SIC = -2*loglik + penalty*k  (on the -2ℓ scale).
      # MSE with reduction='sum' gives Σ(y-ŷ)² ∝ -2ℓ  (nll_scale = 1)
      # BCE/NLL with reduction='sum' gives -ℓ            (nll_scale = 2)
      # Mini-batch covers batch_n < n samples, so scale to full-sample level.
      batch_n <- dim(data)[1]
      nll_scale <- if (SICNN$problem_type == 'regression') 1 else 2
      data_loss_scaled <- (n_train / batch_n) * nll_scale * data_loss
      
      k_smooth <- SICNN$smooth_param_count(epsilon)
      loss <- data_loss_scaled + sic_penalty * k_smooth
      
      # Reporting loss: same scaling but evaluated at final epsilon_T
      k_smooth_T <- SICNN$smooth_param_count(SICNN$sic_epsilon_T)
      loss_report <- (n_train / batch_n) * nll_scale * data_loss$item() + sic_penalty * k_smooth_T$item()
    
      
      if(SICNN$problem_type == 'multiclass classification' | SICNN$problem_type == 'MNIST'){
        prediction <-max.col(output)
        corrects <- corrects + sum(prediction == target)
        totals <- totals + length(target)
        train_loss <- c(train_loss,loss_report)
    
        
        
        
      }
      else if(SICNN$problem_type == 'binary classification'){
        corrects<-corrects + sum((output > 0.5) == target)
        totals <- totals + length(target)
        train_loss <- c(train_loss,loss_report)
        
        
        
        
      }
      else if(SICNN$problem_type == 'custom')
      {
        train_loss <- c(train_loss,loss_report)
      }
      else{#for regression
        train_loss <- c(train_loss,loss_report)
        
      }
      loss$backward()
      opt$step()
      
      
      
    })
    if ( !is.null(scheduler)){sl$step()}
  
    train_acc <- corrects / totals
    if(SICNN$problem_type != 'regression'){
      sic_counts <- SICNN$sic_weight_counts(epsilon = epsilon_report, threshold = sic_threshold)
      density_val <- as.numeric(sic_counts["active"] / sic_counts["total"])
      sparsity_val <- as.numeric(sic_counts["removed"] / sic_counts["total"]) * 100
      message(sprintf(
        "\nEpoch %d, training: loss = %3.5f, acc = %3.5f, density = %3.5f, sparsity = %3.2f%%",
        epoch, mean(train_loss), train_acc, density_val, sparsity_val
      ))
      
      
      accs <- c(accs,train_acc$item())
      losses <- c(losses,mean(train_loss))
    }
    if(SICNN$problem_type == 'regression'){
      sic_counts <- SICNN$sic_weight_counts(epsilon = epsilon_report, threshold = sic_threshold)
      density_val <- as.numeric(sic_counts["active"] / sic_counts["total"])
      sparsity_val <- as.numeric(sic_counts["removed"] / sic_counts["total"]) * 100
      message(sprintf(
        "\nEpoch %d, training: loss = %3.5f, density = %3.5f, sparsity = %3.2f%% \n",
        epoch, mean(train_loss), density_val, sparsity_val
      ))
      
      losses <- c(losses,mean(train_loss))
    }
    sic_counts <- SICNN$sic_weight_counts(epsilon = epsilon_report, threshold = sic_threshold)
    active_weights <- c(active_weights, as.numeric(sic_counts["active"]))
    total_weights <- c(total_weights, as.numeric(sic_counts["total"]))
    removed_weights <- c(removed_weights, as.numeric(sic_counts["removed"]))
    sparsity_pct <- c(sparsity_pct, as.numeric(sic_counts["removed"] / sic_counts["total"]) * 100)
    density <- c(density, as.numeric(sic_counts["active"] / sic_counts["total"]))


    }
    
    l = list('accs' = accs, 'loss' = losses, 'density' = density, 'sparsity_pct' = sparsity_pct, 'active_weights' = active_weights, 'removed_weights' = removed_weights, 'total_weights' = total_weights)
    
    final_run_loss <- mean(train_loss)
    if (final_run_loss < best_loss) {
      best_loss <- final_run_loss
      state_d <- SICNN$state_dict()
      best_state <- lapply(state_d, function(x) x$clone()$detach()$cpu())
      best_l <- l
    }
  }
  
  if (restarts > 1) {
    cat(sprintf("\nLoading best model (loss: %f)\n", best_loss))
    SICNN$load_state_dict(best_state)
    SICNN$to(device = device)
    l <- best_l
  }

  time <- base::proc.time() - start 
  SICNN$elapsed_time <- time[[3]]
  invisible(l)
}

#' @title Validate a trained SICNN model.
#' @description Computes metrics on a validation dataset without computing gradients.
#' Supports model averaging (recommended) by sampling from the variational posterior (\code{num_samples} > 1) 
#' to improve predictions. Returns metrics for both the full model and the sparse model. 
#' @param SICNN An instance of a trained \code{SICNN_Net} to be validated.
#' @param num_samples integer, the number of samples from the variational posterior to be used for model averaging.
#' @param test_dl An instance of \code{torch::dataloader}, containing the validation data.
#' @param device The device to perform validation on. Default is 'cpu'; other options include 'gpu' and 'mps'.
#' @return A list containing the following elements:
#'   \describe{
#'     \item{accuracy_full_model}{Classification accuracy of the full (dense) model (if classification).}
#'     \item{accuracy_sparse}{Classification accuracy using only weights in active paths (if classification).}
#'     \item{validation_error}{Root mean squared error for the full model (if regression).}
#'     \item{validation_error_sparse}{Root mean squared error using only weights in active paths (if regression).}
#'     \item{density}{Proportion of weights with posterior inclusion probability > 0.5 in the whole network.}
#'     \item{density_active_path}{Proportion of weights with inclusion probability > 0.5 after removing weights not in active paths.}
#'   }     
#' @export
validate_SICNN <- function(SICNN,num_samples,test_dl,device = 'cpu'){
  SICNN$eval()
  sparse_sic <- (!is.null(SICNN$criterion_trained) && SICNN$criterion_trained == "SIC")
  # Deterministic SIC: multiple samples are redundant.
  if(sparse_sic){ num_samples <- 1 }
  corrects <- 0
  corrects_sparse <-0
  totals <- 0 
  val_loss <- c()
  val_loss_mpm <-c()
  out_shape <- 1 #if binary classification or regression
  if (SICNN$input_skip) {
    if (!is.null(SICNN$criterion_trained) && SICNN$criterion_trained == "SIC") {
      SICNN$compute_paths_input_skip(epsilon = SICNN$sic_epsilon_T, threshold = SICNN$sic_threshold)
    } else {
      SICNN$compute_paths_input_skip()
    }
  } else {
    if (!is.null(SICNN$criterion_trained) && SICNN$criterion_trained == "SIC") {
      SICNN$compute_paths(epsilon = SICNN$sic_epsilon_T, threshold = SICNN$sic_threshold)
    } else {
      SICNN$compute_paths()
    }
  }
  SICNN$computed_paths <- TRUE
  torch::with_no_grad({ 
    coro::loop(for (b in test_dl){
      target <- b[[2]]$to(device=device)
      if(SICNN$problem_type == 'multiclass classification'| SICNN$problem_type == 'MNIST'){ #nll loss needs float tensors but bce loss needs long tensors 
        target <- torch::torch_tensor(target,dtype = torch::torch_long())
        out_shape <- max(target)$item()
      }
      outputs <- torch::torch_zeros(num_samples,dim(b[[1]])[1],out_shape)$to(device=device)
      output_mpm <- torch::torch_zeros_like(outputs)
      for(i in 1:num_samples){
        data <- b[[1]]$to(device = device)
        outputs[i]<- SICNN(data,sparse=FALSE)
        output_mpm[i] <- SICNN(data,sparse=TRUE)
        
      }
      out_full <-outputs$mean(1) #average over num_samples dimension
      out_mpm <-output_mpm$mean(1)
      
      if(SICNN$problem_type == 'multiclass classification' | SICNN$problem_type == 'MNIST'){
        prediction <-max.col(out_full)
        corrects <- corrects + sum(prediction == target)
        totals <- totals + length(target)
        
        #prediction using only weights in active paths
        prediction_mpm <-max.col(out_mpm)
        corrects_sparse <- corrects_sparse + sum(prediction_mpm == target)
        
        
      }
      
      else if(SICNN$problem_type == 'binary classification'){
        out_full <- out_full$squeeze()
        out_mpm <-out_mpm$squeeze()
        corrects<-corrects + sum((out_full > 0.5) == target)
        corrects_sparse<-corrects_sparse + sum((out_mpm > 0.5) == target)
        totals <- totals + length(target)
      }
      else{#for regression
        out_full <- out_full$squeeze()
        out_mpm <-out_mpm$squeeze()
        
        loss <- torch::torch_sqrt(torch::nnf_mse_loss(out_full, target))
        loss_mpm <- torch::torch_sqrt(torch::nnf_mse_loss(out_mpm, target))
        val_loss <- c(val_loss,loss$item())
        val_loss_mpm <- c(val_loss_mpm,loss_mpm$item())
      }
      
      
      
      
    })  
  })
  acc_full<- corrects / totals
  acc_sparse <- corrects_sparse / totals
  
  # Density reporting depends on how the model was trained.
  if(!is.null(SICNN$criterion_trained) && SICNN$criterion_trained == "SIC"){
    epsilon_used <- if(!is.null(SICNN$sic_epsilon_T)) SICNN$sic_epsilon_T else 1e-5
    thr_used <- if(!is.null(SICNN$sic_threshold)) SICNN$sic_threshold else 0.5
    density <- SICNN$sic_density(epsilon = epsilon_used, threshold = thr_used)
    density2 <- SICNN$sic_density_active_path(epsilon = epsilon_used, threshold = thr_used)
    sic_counts <- SICNN$sic_weight_counts(epsilon = epsilon_used, threshold = thr_used)
    sparsity_pct <- as.numeric(sic_counts["removed"] / sic_counts["total"]) * 100
    active_weights <- as.numeric(sic_counts["active"])
    removed_weights <- as.numeric(sic_counts["removed"])
    total_weights <- as.numeric(sic_counts["total"])
  } else {
    density <- SICNN$density()
    density2 <- SICNN$density_active_path()
  }
  if(SICNN$problem_type!='regression'){
    l = list('accuracy_full_model' = acc_full$item(),
             'accuracy_sparse' = acc_sparse$item(),
             'density'=density,
             'density_active_path'=density2)
    if(!is.null(SICNN$criterion_trained) && SICNN$criterion_trained == "SIC"){
      l$sparsity_pct <- sparsity_pct
      l$active_weights <- active_weights
      l$removed_weights <- removed_weights
      l$total_weights <- total_weights
    }
  }
  else{
    l = list('validation_error'=mean(val_loss),
             'validation_error_sparse' = mean(val_loss_mpm),
             'density'=density,
             'density_active_path'=density2)
    if(!is.null(SICNN$criterion_trained) && SICNN$criterion_trained == "SIC"){
      l$sparsity_pct <- sparsity_pct
      l$active_weights <- active_weights
      l$removed_weights <- removed_weights
      l$total_weights <- total_weights
    }
  }
  return(l)
}





