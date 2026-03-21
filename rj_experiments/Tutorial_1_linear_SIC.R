library(SICNN)
#### Tutorial 1 (SIC): simulated data with linear effects using smooth BIC criterion

i <- 1000
j <- 15

set.seed(42)
torch::torch_manual_seed(42)

# generate some data
X <- matrix(rnorm(i * j, mean = 0, sd = 1), ncol = j)
# make some X relevant for prediction
y_base <- 0.6 * X[, 1] - 0.4 * X[, 2] + 0.5 * X[, 3] + rnorm(n = i, sd = 0.1)
sim_data <- as.data.frame(X)
sim_data <- cbind(sim_data, y_base)

loaders <- get_dataloaders(
  sim_data,
  train_proportion = 0.9,
  train_batch_size = 450,
  test_batch_size = 100,
  standardize = FALSE
)
train_loader <- loaders$train_loader
test_loader  <- loaders$test_loader

problem <- "regression"
sizes <- c(j, 5, 5, 1) # 2 hidden layers, 5 neurons in each 
incl_priors <- c(0.5, 0.5, 0.5) # prior inclusion probability
stds <- c(1, 1, 1) # prior for the standard deviation of the weights
incl_inits <- matrix(rep(c(-10, 10), 3), nrow = 2, ncol = 3) # inclusion inits
device <- "cpu" # can also be 'gpu' or 'mps'

model_input_skip <- SICNN_Net(
  problem_type = problem,
  sizes = sizes,
  input_skip = TRUE,
  device = device
)

# train using smooth BIC / SIC criterion with epsilon-telescope
results_sic <- train_SICNN(
  epochs    = 2000,
  restarts  = 1,
  SICNN     = model_input_skip,
  lr        = 0.002,
  train_dl  = train_loader,
  device    = device,
  scheduler = "step",
  sch_step_size = 500,
  n_train   = i,
  epsilon_1 = 1,
  epsilon_T = 1e-5,
  steps_T   = 200,
  sic_threshold = 0.5
)

validate_SICNN(
  SICNN     = model_input_skip,
  num_samples = 10,
  test_dl   = test_loader,
  device    = device
)

coef(
  model_input_skip,
  dataset      = train_loader,
  inds         = c(1, 2, 5, 10, 20),
  output_neuron = 1,
  num_data     = 5,
  num_samples  = 10
)

plot(model_input_skip)

x <- train_loader$dataset$tensors[[1]] # grab the dataset
y <- train_loader$dataset$tensors[[2]] 
ind <- 58
data <- x[ind, ] # plot this specific data-point
output <- y[ind]
print(output$item())
plot(model_input_skip, type = "local", data = data)

summary(model_input_skip, criterion = "SIC", epsilon = 1e-5, threshold = 0.5)

