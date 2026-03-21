# KMNIST Convolutional Architecture Tutorial using SICNN
# This script trains a deep convolutional SICNN network on the KMNIST dataset.

library(torch)
library(torchvision)
library(SICNN)

# ---------------------------------------------------------
# 1. Device Configuration
# ---------------------------------------------------------
device <- "cpu"
cat(sprintf("Using device: %s\n", device))

# ---------------------------------------------------------
# 2. Download and Prepare Data
# ---------------------------------------------------------
dir <- "./dataset/kmnist"
cat("Downloading and reading KMNIST dataset...\n")
train_ds <- torchvision::kmnist_dataset(
  dir,
  download = TRUE,
  transform = torchvision::transform_to_tensor
)

test_ds <- torchvision::kmnist_dataset(
  dir,
  train = FALSE,
  transform = torchvision::transform_to_tensor
)

train_loader <- dataloader(train_ds, batch_size = 100, shuffle = TRUE)
test_loader <- dataloader(test_ds, batch_size = 100)

# ---------------------------------------------------------
# 3. Define the Architecture
# ---------------------------------------------------------
torch_manual_seed(42)

# Image size 28x28 -> conv1(kernel=5) -> 24x24 -> maxpool(2) -> 12x12
# 12x12 -> conv2(kernel=5) -> 8x8 -> maxpool(2) -> 4x4
# Flattened size: 64 channels * 4 * 4 = 1024
conv_layer_1 <- SICNN_Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, device = device)
conv_layer_2 <- SICNN_Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, device = device)
linear_layer_1 <- SICNN_Linear(in_features = 1024, out_features = 300, device = device, bias = FALSE)
linear_layer_2 <- SICNN_Linear(in_features = 300, out_features = 10, device = device, bias = FALSE)

model <- SICNN_ConvNet(conv1 = conv_layer_1, 
                       conv2 = conv_layer_2, 
                       fc1 = linear_layer_1, 
                       fc2 = linear_layer_2, 
                       device = device)
model$to(device = device)

# ---------------------------------------------------------
# 4. Train the Model
# ---------------------------------------------------------
n_epochs <- 50
cat("Starting training for", n_epochs, "epochs...\n")

res <- train_SICNN(
  epochs = n_epochs,
  SICNN = model,
  lr = 0.001,
  train_dl = train_loader,
  n_train = length(train_ds),
  device = device
)

# ---------------------------------------------------------
# 5. Evaluate and Visualize
# ---------------------------------------------------------
cat("\nTraining complete. Running validation on the test set...\n")
validate_SICNN(model, num_samples = 10, test_dl = test_loader, device = device)

plot(model)

# Plot Loss & Sparsity
par(mfrow=c(1,2))
plot(res$loss, type='l', col='blue', lwd=2, 
     main="Training Loss", xlab="Epoch", ylab="Scaled Negative Log-Likelihood")
plot(res$sparsity_pct, type='l', col='red', lwd=2, ylim=c(0, 100),
     main="Sparsity (%)", xlab="Epoch", ylab="Active Weights Removed (%)")
