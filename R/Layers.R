library(torch)

#' @title Class to generate an SICNN feed forward layer
#' @description This module implements a fully connected SICNN layer.
#' @param in_features integer, number of input neurons.
#' @param out_features integer, number of output neurons.
#' @param device The device to be used. Default is CPU.
#' @param bias logical, whether to use bias.
#' @keywords internal
#' @export
SICNN_Linear <- torch::nn_module(
  "SICNN_Linear",
  initialize = function(in_features, out_features, device = "cpu", bias = TRUE, conv_net = FALSE) {
    self$in_features <- in_features
    self$out_features <- out_features
    self$device <- device
    self$has_bias <- bias
    
    self$weight_mean <- torch::nn_parameter(torch::torch_empty(out_features, in_features, device = device))
    if (self$has_bias) {
      self$bias_mean <- torch::nn_parameter(torch::torch_empty(out_features, device = device))
    } else {
      self$bias_mean <- NULL
    }
    
    self$alpha_active_path <- torch::torch_ones(out_features, in_features, device = device)
    
    self$reset_parameters()
  },
  reset_parameters = function() {
    stdv <- 1 / sqrt(self$in_features)
    torch::nn_init_uniform_(self$weight_mean, -stdv, stdv)
    if (self$has_bias) {
      torch::nn_init_uniform_(self$bias_mean, -stdv, stdv)
    }
  },
  forward = function(input, sparse=FALSE) {
    w <- self$weight_mean
    if (sparse){
       weight <- w * self$alpha_active_path
    } else {
       weight <- w
    }
    bias <- self$bias_mean
    activations <- torch::torch_matmul(input, torch::torch_t(weight))
    if (self$has_bias) activations <- activations + bias
    return(activations)
  }
)

#' @title Class to generate an SICNN convolutional layer
#' @description This module implements an SICNN Conv2d layer.
#' @param in_channels integer.
#' @param out_channels integer.
#' @param kernel_size integer or tuple.
#' @param stride integer or tuple.
#' @param padding integer, tuple or string.
#' @param dilation integer or tuple.
#' @param device The device to be used. Default is CPU.
#' @param bias logical.
#' @keywords internal
#' @export
SICNN_Conv2d <- torch::nn_module(
  "SICNN_Conv2d",
  initialize = function(in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, device = "cpu", bias = TRUE) {
    self$in_channels <- in_channels
    self$out_channels <- out_channels
    self$kernel_size <- if(is.numeric(kernel_size) && length(kernel_size)==1) c(kernel_size, kernel_size) else kernel_size
    self$stride <- stride
    self$padding <- padding
    self$dilation <- dilation
    self$device <- device
    self$has_bias <- bias
    
    self$weight_mean <- torch::nn_parameter(torch::torch_empty(out_channels, in_channels, self$kernel_size[1], self$kernel_size[2], device = device))
    if (self$has_bias) {
      self$bias_mean <- torch::nn_parameter(torch::torch_empty(out_channels, device = device))
    } else {
      self$bias_mean <- NULL
    }
    
    self$alpha_active_path <- torch::torch_ones(out_channels, in_channels, self$kernel_size[1], self$kernel_size[2], device = device)
    
    self$reset_parameters()
  },
  reset_parameters = function() {
    n <- self$in_channels * self$kernel_size[1] * self$kernel_size[2]
    stdv <- 1 / sqrt(n)
    torch::nn_init_uniform_(self$weight_mean, -stdv, stdv)
    if (self$has_bias) {
      torch::nn_init_uniform_(self$bias_mean, -stdv, stdv)
    }
  },
  forward = function(input, sparse=FALSE) {
    w <- self$weight_mean
    if(sparse){
      weight <- w * self$alpha_active_path
    } else {
      weight <- w
    }
    bias <- self$bias_mean
    activations <- torch::nnf_conv2d(input=input, weight=weight, bias=bias, stride=self$stride, padding=self$padding, dilation=self$dilation)
    return(activations)
  }
)
