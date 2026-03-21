test_that("Smoke: tiny model trains one epoch", {
  testthat::skip_on_cran()
  if (!requireNamespace("torch", quietly = TRUE)) {
    testthat::skip("torch not available")
  }
  
  i = 5000
  j = 15
  
  set.seed(2)
  torch::torch_manual_seed(2)
  #generate some data
  X <- matrix(rnorm(i*j,mean =-0.1 ,sd = 0.1), ncol = j)
  
  #make some X relevant for prediction
  y_base <-  0.1 * log(abs(X[,1])) + 3 * cos(X[,2]) + 2* X[,3] * X[,4] + 2 * X[,5] - 2* X[,6] **2 + rnorm(i,sd = 0.01) 
  y <- c()
  # change y to 0 and 1
  y[y_base > median(y_base)] = 1
  y[y_base <= median(y_base)] = 0
  
  
  sim_data <- as.data.frame(X)
  sim_data <-cbind(sim_data,y)
  
  
  
  
  loaders <- get_dataloaders(sim_data,train_proportion = 0.9,
                             train_batch_size = 1500,test_batch_size = 500,
                             standardize = FALSE)
  train_loader <- loaders$train_loader
  test_loader  <- loaders$test_loader
  
  problem <- 'binary classification'
  sizes <- c(j,5,5,1) # 2 hidden layers, 5 neurons in each 
  device <- 'cpu' #can also be 'gpu' or 'mps'
  
  
  model_input_skip <- SICNN_Net(problem_type = problem,sizes = sizes,
                                input_skip = TRUE,device = device)
  
  
  
  res <- train_SICNN(epochs = 1,SICNN = model_input_skip,
              lr = 0.005,train_dl = train_loader,n_train = 5000)
  
  validate_SICNN(SICNN = model_input_skip,num_samples = 1,test_dl = test_loader,device)
  
  expect_true(length(res$loss) == 1)
  expect_true(is.numeric(res$density[1]))
})



