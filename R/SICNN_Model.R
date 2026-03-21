library(torch)

#' @title Feed-forward Latent Binary Bayesian Neural Network (SICNN)
#' @description
#' Each layer is defined by \code{SICNN_Linear}. 
#' For example, \code{sizes = c(20, 200, 200, 5)} generates a network with:
#' \itemize{
#'   \item 20 input features,
#'   \item two hidden layers of 200 neurons each,
#'   \item an output layer with 5 neurons.
#'}
#' @param problem_type character, one of:
#' \code{'binary classification'}, \code{'multiclass classification'}, \code{'regression'}, or \code{'custom'}.
#' @param sizes Integer vector specifying the layer sizes of the network. The first element is the input size,
#' the last is the output size, and the intermediate integers represent hidden layers.
#' @param input_skip logical, whether to include input_skip.
#' @param device the device to be trained on. Can be 'cpu', 'gpu' or 'mps'. Default is cpu.
#' @param raw_output logical, whether the network skips the last sigmoid/softmax layer to compute local explanations.
#' @param custom_act Allows the user to submit their own customized activation function.
#' @param link User can define their own link function (not implemented yet).
#' @param nll User can define their own likelihood function (not implemented yet).
#' @param bias logical, whether to include bias terms in linear layers. Default is TRUE.
#' @examples
#' \donttest{
#' layers <- c(10,2,5) 
#' prob <- 'multiclass classification'
#' net <- SICNN_Net(problem_type = prob, sizes = layers, input_skip = FALSE, device = 'cpu')
#' x <- torch::torch_rand(20,10,requires_grad = FALSE)
#' output <- net(x) 
#' net$sic_density(epsilon=1e-5, threshold=0.5) 
#' }
#' @return A \code{torch::nn_module} object representing the SICNN. 
#'   It includes the following methods:
#'   \itemize{
#'     \item \code{forward(x, sparse = FALSE)}: Performs a forward pass through the whole network.
#'     \item \code{kl_div()}: Returns the KL divergence of the network.
#'     \item \code{density()}: Returns the density of the whole network, i.e. the proportion of weights
#'     with inclusion probabilities greater than 0.5.
#'     \item \code{compute_paths()}: Computes active paths through the network without input-skip. 
#'     \item \code{compute_paths_input_skip()}: Computes active paths with  input-skip enabled. 
#'     \item \code{density_active_path()}: Returns network density after removing inactive paths.
#'     \item \code{smooth_param_count(epsilon)}: Returns the smooth L0-based effective parameter
#'     count used in the smooth information criterion (SIC) of O’Neill and Burke (2023),
#'     based on the penalty \eqn{\phi_\epsilon(x) = x^2 / (x^2 + \epsilon^2)} applied to the
#'     layer weight means.
#'   }
#'
#' @export
SICNN_Net <- torch::nn_module(
  "SICNN_Net",
  
  initialize = function(problem_type, sizes, input_skip = FALSE,
                        device = 'cpu', raw_output = FALSE, custom_act = NULL,
                        link = NULL, nll = NULL, bias = TRUE) {
    self$device <- device
    self$layers <- torch::nn_module_list()
    self$problem_type <- problem_type
    self$input_skip <- input_skip
    self$bias <- bias
    self$sizes <- sizes
    self$elapsed_time <- 0
    self$raw_output <- raw_output # TRUE when we want to compute local explanations
    self$act <- torch::nn_leaky_relu(0.00)
    self$computed_paths <- FALSE
    if(! is.null(custom_act)){
      self$act <- custom_act
    }
     #define the layers of the network
     #for the first layer input-skip and SICNNs are the same,
     #but for all subsequent layers, input-skip will have shape (dim + p,dim)
     #where p is the number of variables. (i.e. the first element of sizes)
     for(i in 1:(length(sizes)-2)){
      if(i == 1 || input_skip == FALSE){in_shape <- sizes[i]} #no input skip on the first layer or with std SICNNs
      else(in_shape <- sizes[i] + sizes[1])
      self$layers$append(SICNN_Linear(
        in_shape,
        sizes[i+1],
        device=self$device,
        bias=self$bias))
    }
    if(input_skip){out_size <- sizes[length(sizes) - 1] + sizes[1]
    }else{out_size <- sizes[length(sizes) - 1]}
    self$out_layer <- (SICNN_Linear(
      out_size,
      sizes[length(sizes)],
      device=self$device,
      bias=self$bias))


    if(problem_type == 'binary classification'){
      self$out <- torch::nn_sigmoid()
      self$loss_fn <- torch::nn_bce_loss(reduction='sum')
    }
    else if(problem_type == 'multiclass classification' | problem_type == 'MNIST'){
      self$out <- torch::nn_log_softmax(dim = 2)
      self$loss_fn <- torch::nn_nll_loss(reduction='sum')
    }
    else if(problem_type == 'regression'){
      self$out <- torch::nn_identity()
      self$loss_fn <- torch::nn_mse_loss(reduction='sum')
    }else if(problem_type == 'custom')
    {
      if(length(link) == 0 | length(nll) == 0)
        stop("Under custom problem, link function and the negative log likelihood must be provided as torch functions")
      self$out <- link()
      self$loss_fn <- nll(reduction='sum')
    }
    else(stop('the type of problem must either be: \'binary classification\', 
              \'multiclass classification\', \'regression\' or \'custom\''))
  },
  forward = function(x, sparse = FALSE){
    if(self$problem_type == 'MNIST')(x <- x$view(c(-1,28*28)))
    #regular SICNN
    if(!self$input_skip){
      x <- x$view(c(-1,self$sizes[1]))
      for(l in self$layers$children){
       x <- self$act(l(x,sparse=sparse)) #iterate over hidden layers
        
      }
      x <- self$out(self$out_layer(x,sparse=sparse))
    }
    else{x_input <- x$view(c(-1,self$sizes[1]))
    x <- self$layers$children$`0`(x_input,sparse=sparse) #first layer
    j <- 1
    for(l in self$layers$children){
      if(j > 1){#skip the first layer when iterating.
        x <- l(torch::torch_cat(c(x,x_input),dim = 2),sparse=sparse)
      }
      j <- j + 1
    }
    
    #if we only want the raw output, skip sigmoid/softmax
    if(self$raw_output){
      x<- self$out_layer(torch::torch_cat(c(x,x_input),dim = 2),sparse=sparse)
    }
    else(x<- self$out(self$out_layer(torch::torch_cat(c(x,x_input),dim = 2),sparse=sparse)))

      
    }
   
    return(x)
    },
  smooth_param_count = function(epsilon){
    # Smooth approximation to the number of effective parameters used in SIC.
    # In this architecture, the effective coefficient for each edge is
    #   w_eff = weight_mean * alpha_soft
    # where alpha_soft = sigmoid(lambda_l). We apply the smooth L0 surrogate
    #  phi_epsilon(w) = w^2 / (w^2 + epsilon^2)
    # to w_eff, consistent with the coefficient being regularized.
    if(missing(epsilon) || !is.numeric(epsilon) || length(epsilon) != 1 || epsilon <= 0){
      stop("epsilon must be a positive numeric scalar")
    }
    smooth_l0 <- function(tensor, epsilon){
      num <- tensor ^ 2
      den <- tensor ^ 2 + epsilon ^ 2
      torch::torch_sum(num / den)
    }
    k_smooth <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$device)
    # hidden layers
    for(l in self$layers$children){
      w_eff <- l$weight_mean
      k_smooth <- k_smooth + smooth_l0(w_eff, epsilon)
    }
    # output layer
    w_eff_out <- self$out_layer$weight_mean
    k_smooth <- k_smooth + smooth_l0(w_eff_out, epsilon)
    return(k_smooth)
  },

  # Overall SIC density (proportion of active edges) based on w_eff.
  sic_density = function(epsilon, threshold = 0.5, threshold_type = "phi"){
    if(missing(epsilon) || !is.numeric(epsilon) || length(epsilon) != 1 || epsilon <= 0){
      stop("epsilon must be a positive numeric scalar")
    }
    if(!is.numeric(threshold) || length(threshold) != 1 || threshold <= 0){
      stop("threshold must be a positive numeric scalar")
    }
    phi_active <- function(w_eff){
      if(threshold_type == "abs"){
        torch::torch_abs(w_eff) > threshold
      } else {
        w2 <- w_eff ^ 2
        w2 / (w2 + epsilon ^ 2) > threshold
      }
    }
    num_incl <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$device)
    tot <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$device)
    for(l in self$layers$children){
      w_eff <- l$weight_mean
      m <- phi_active(w_eff)
      num_incl <- num_incl + torch::torch_sum(m$to(torch::torch_float32()))
      tot <- tot + m$numel()
    }
    w_eff_out <- self$out_layer$weight_mean
    m_out <- phi_active(w_eff_out)
    num_incl <- num_incl + torch::torch_sum(m_out$to(torch::torch_float32()))
    tot <- tot + m_out$numel()
    return((num_incl / tot)$item())
  },

  # SIC density restricted to edges on the active paths (alpha_active_path mask).
  sic_density_active_path = function(epsilon, threshold = 0.5, threshold_type = "phi"){
    if(missing(epsilon) || !is.numeric(epsilon) || length(epsilon) != 1 || epsilon <= 0){
      stop("epsilon must be a positive numeric scalar")
    }
    if(!is.numeric(threshold) || length(threshold) != 1 || threshold <= 0){
      stop("threshold must be a positive numeric scalar")
    }
    if(self$computed_paths == FALSE){
      if(self$input_skip){self$compute_paths_input_skip()} else {self$compute_paths()}
    }
    phi_active <- function(w_eff){
      if(threshold_type == "abs"){
        torch::torch_abs(w_eff) > threshold
      } else {
        w2 <- w_eff ^ 2
        w2 / (w2 + epsilon ^ 2) > threshold
      }
    }
    num_incl <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$device)
    tot <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$device)
    for(l in self$layers$children){
      alpha_mask_hard <- l$alpha_active_path
      w_eff <- l$weight_mean
      m_phi <- phi_active(w_eff)
      m_on_path <- (alpha_mask_hard > 0) & m_phi
      num_incl <- num_incl + torch::torch_sum(m_on_path$to(torch::torch_float32()))
      tot <- tot + alpha_mask_hard$numel()
    }
    alpha_mask_out_hard <- self$out_layer$alpha_active_path
    w_eff_out <- self$out_layer$weight_mean
    m_phi_out <- phi_active(w_eff_out)
    m_on_path_out <- (alpha_mask_out_hard > 0) & m_phi_out
    num_incl <- num_incl + torch::torch_sum(m_on_path_out$to(torch::torch_float32()))
    tot <- tot + alpha_mask_out_hard$numel()
    return((num_incl / tot)$item())
  },

  # Counts active/removed edges under SIC.
  sic_weight_counts = function(epsilon, threshold = 0.5, threshold_type = "phi"){
    if(missing(epsilon) || !is.numeric(epsilon) || length(epsilon) != 1 || epsilon <= 0){
      stop("epsilon must be a positive numeric scalar")
    }
    if(!is.numeric(threshold) || length(threshold) != 1 || threshold <= 0){
      stop("threshold must be a positive numeric scalar")
    }
    phi_active <- function(w_eff){
      if(threshold_type == "abs"){
        torch::torch_abs(w_eff) > threshold
      } else {
        w2 <- w_eff ^ 2
        w2 / (w2 + epsilon ^ 2) > threshold
      }
    }
    num_incl <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$device)
    tot <- torch::torch_tensor(0, dtype = torch::torch_float32(), device = self$device)
    for(l in self$layers$children){
      w_eff <- l$weight_mean
      m <- phi_active(w_eff)
      num_incl <- num_incl + torch::torch_sum(m$to(torch::torch_float32()))
      tot <- tot + m$numel()
    }
    w_eff_out <- self$out_layer$weight_mean
    m_out <- phi_active(w_eff_out)
    num_incl <- num_incl + torch::torch_sum(m_out$to(torch::torch_float32()))
    tot <- tot + m_out$numel()
    active <- num_incl$item()
    total <- tot$item()
    removed <- total - active
    return(c(active = active, total = total, removed = removed))
  },
  compute_paths = function(epsilon = NULL, threshold = NULL, threshold_type = NULL){
    if(self$input_skip == TRUE){
      stop('self$input_skip must be FALSE to use this function')
    }
    if (is.null(epsilon)) epsilon <- self$sic_epsilon_T
    if (is.null(threshold)) threshold <- self$sic_threshold
    if (is.null(threshold_type)) threshold_type <- if (!is.null(self$sic_report_threshold_type)) self$sic_report_threshold_type else "phi"
    self$computed_paths <- TRUE
    # sending an input through the network of alpha matrices (0 and 1)
    #and then backpropagating to find active paths
    a <- rep(1, times = self$layers$children$`0`$weight_mean$shape[2])
    x0 <- torch::torch_tensor(a, dtype = torch::torch_float32(),device = self$device)
    alpha_mats <- list()
    
    for(l in self$layers$children){
      w_eff <- l$weight_mean$detach()
      if(threshold_type == "abs"){
        alpha <- (torch::torch_abs(w_eff) > threshold) * 1
      } else {
        phi <- w_eff^2 / (w_eff^2 + epsilon^2)
        alpha <- (phi > threshold) * 1
      }
      alpha$requires_grad = TRUE
      alpha_mats<- append(alpha_mats,alpha)
      x0 <- torch::torch_matmul(x0, torch::torch_t(alpha))
    }
    w_eff_out <- self$out_layer$weight_mean$detach()
    if(threshold_type == "abs"){
        alpha_out <- ((torch::torch_abs(w_eff_out) > threshold) * 1)$detach()
    } else {
        phi_out <- w_eff_out^2 / (w_eff_out^2 + epsilon^2)
        alpha_out <- ((phi_out > threshold) * 1)$detach()
    }
    alpha_out$requires_grad = TRUE
    alpha_mats <-append(alpha_mats,alpha_out)
    x0 <- torch::torch_matmul(x0, torch::torch_t(alpha_out))
    L <- torch::torch_sum(x0) #summing in case more than 1 output. This is
    #equivalent to backpropagate for each output node.
    L$backward() #compute derivatives to get active paths
                  #any alpha preceding an alpha with value 0 will also become
                  #zero when gradients are passed backwards, and thus we will
                  #be left with the active paths.
    i <- 1
    alpha_mats_out <- list()
    for(j in self$layers$children){
      alp = alpha_mats[[i]] * alpha_mats[[i]]$grad
      alp[alp!=0] <- 1
      alpha_mats_out<- append(alpha_mats_out,alp$detach())
      j$alpha_active_path <- alp$detach()
      i = i +1
      
    }
    alp_out <- alpha_mats[[length(alpha_mats)]] *alpha_mats[[length(alpha_mats)]]$grad #last alpha
    alp_out[alp_out!=0] <- 1
    alpha_mats_out<- append(alpha_mats_out,alp_out$detach())
    
    
    self$out_layer$alpha_active_path <- alp_out$detach()
    
   
    
    return(alpha_mats_out)
    
  },
  compute_paths_input_skip = function(epsilon = NULL, threshold = NULL, threshold_type = NULL){
    if(self$input_skip == FALSE){
      stop('self$input_skip must be TRUE to use this funciton')
    }
    if (is.null(epsilon)) epsilon <- self$sic_epsilon_T
    if (is.null(threshold)) threshold <- self$sic_threshold
    if (is.null(threshold_type)) threshold_type <- if (!is.null(self$sic_report_threshold_type)) self$sic_report_threshold_type else "phi"
    self$computed_paths <- TRUE
    # sending an input through the network of alpha matrices (0 and 1)
    #and then backpropagating to find active paths
    a <- rep(1, times = self$layers$children$`0`$weight_mean$shape[2])
    x0 <- torch::torch_tensor(a, dtype = torch::torch_float32(),device = self$device)$unsqueeze(dim = 1)
    alpha_mats <- list()
    
    w_eff_input <- self$layers$children$`0`$weight_mean$detach()
    if(threshold_type == "abs"){
      alpha_input <- (torch::torch_abs(w_eff_input) > threshold) * 1
    } else {
      phi_input <- w_eff_input^2 / (w_eff_input^2 + epsilon^2)
      alpha_input <- (phi_input > threshold) * 1
    }
    alpha_input$requires_grad = TRUE
    alpha_mats <- append(alpha_mats,alpha_input)
    
    x<- torch::torch_matmul(x0, torch::torch_t(alpha_input))
    
    j <- 1
    for(l in self$layers$children){
      if(j > 1){
        x <- (torch::torch_cat(c(x,x0),dim = 2))
        w_eff <- l$weight_mean$detach()
        if(threshold_type == "abs"){
          alpha <- (torch::torch_abs(w_eff) > threshold) * 1
        } else {
          phi <- w_eff^2 / (w_eff^2 + epsilon^2)
          alpha <- (phi > threshold) * 1
        }
        alpha$requires_grad = TRUE
        alpha_mats<- append(alpha_mats,alpha)
        x <- torch::torch_matmul(x, torch::torch_t(alpha))
      }
      j <- j + 1
    }
    #output layer
    w_eff_out <- self$out_layer$weight_mean$detach()
    if(threshold_type == "abs"){
      alpha_out <- ((torch::torch_abs(w_eff_out) > threshold) * 1)$detach()
    } else {
      phi_out <- w_eff_out^2 / (w_eff_out^2 + epsilon^2)
      alpha_out <- ((phi_out > threshold) * 1)$detach()
    }
    alpha_out$requires_grad = TRUE
    alpha_mats <-append(alpha_mats,alpha_out)
    x_out <- (torch::torch_cat(c(x,x0),dim = 2))
    x_out <- torch::torch_matmul(x_out, torch::torch_t(alpha_out))
    
    L <- torch::torch_sum(x_out) #summing in case more than 1 output. This is
    #equivalent to backpropagate for each output node.
    L$backward() #compute derivatives to get active paths
    #any alpha preceding an alpha with value 0 will also become
    #zero when gradients are passed backwards, and thus we will
    #be left with the active paths.
    i <- 1
    alpha_mats_out <- list()
    for(j in self$layers$children){
      alp = alpha_mats[[i]] * alpha_mats[[i]]$grad
      alp[alp!=0] <- 1
      alpha_mats_out<- append(alpha_mats_out,alp$detach())
      j$alpha_active_path <- alp$detach()
      i = i +1
      
    }
    alp_out <- alpha_mats[[length(alpha_mats)]] *alpha_mats[[length(alpha_mats)]]$grad #last alpha
    alp_out[alp_out!=0] <- 1
    alpha_mats_out<- append(alpha_mats_out,alp_out$detach())
    
    
    self$out_layer$alpha_active_path <- alp_out$detach()
    
    
    
    
    return(alpha_mats_out)
    
  },

  
  
  density_active_path = function(){
    thr_t <- if(!is.null(self$sic_report_threshold_type)) self$sic_report_threshold_type else "phi"
    return(self$sic_density_active_path(self$sic_epsilon_T, self$sic_threshold, thr_t))
  }
  
)











