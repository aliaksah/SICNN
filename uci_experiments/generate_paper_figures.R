library(SICNN)
library(torch)
library(stats)

# Ensure the figures directory exists
dir.create("paper/figures", showWarnings = FALSE, recursive = TRUE)

# --- Figure 1: Tutorial 1 (Linear Regression) ---
# n=1000, p=15, 3 active features.
set.seed(42)
n <- 1000
p <- 15
x <- matrix(rnorm(n * p), n, p)
b <- c(1, -2, 1.5, rep(0, p - 3))
y <- x %*% b + rnorm(n, 0, 0.5)
df1 <- as.data.frame(cbind(x, y))
colnames(df1)[ncol(df1)] <- "target"

# Scaled data
for(i in seq_len(p)) {
  r <- range(df1[[i]])
  df1[[i]] <- (df1[[i]] - r[1]) / (r[2] - r[1])
}
target_r <- range(df1$target)
df1$target <- (df1$target - target_r[1]) / (target_r[2] - target_r[1])

loaders1 <- get_dataloaders(df1, train_proportion = 0.8, train_batch_size = 32, test_batch_size = 200)
model1 <- SICNN_Net(problem_type = "regression", sizes = c(p, 5, 5, 1), input_skip = TRUE, device = "cpu")
train_SICNN(epochs=2000, restarts=1, SICNN=model1, lr=0.01, train_dl=loaders1$train_loader, n_train=800,
           epsilon_1=1, epsilon_T=1e-5, steps_T=1600, sic_threshold=0.5, penalty=log(800))

# Save PDF
pdf("paper/figures/tut1_local.pdf", width=6, height=4)
# Take a sample point for local plot
sample_x <- torch::dataloader_next(torch::dataloader_make_iter(loaders1$train_loader))[[1]]
plot(model1, type="local", data=sample_x[1,])
dev.off()
cat("Saved paper/figures/tut1_local.pdf\n")

# --- Figure 2: Tutorial 2 (Non-linear Classification) ---
# n=1000, p=15, 6 active features.
# Let's use a simpler non-linear setup if needed, but let's follow the text: log, cos, interactions.
set.seed(43)
x2 <- matrix(runif(n * p, 0.1, 1), n, p)
# f = log(x1)*x2 + cos(x3*x4) + x5^x6
f <- log(x2[,1])*x2[,2] + cos(x2[,3]*x2[,4]) + x2[,5]^x2[,6]
y2 <- ifelse(f > median(f), 1, 0)
df2 <- as.data.frame(cbind(x2, y2))
colnames(df2)[ncol(df2)] <- "target"

loaders2 <- get_dataloaders(df2, train_proportion = 0.8, train_batch_size = 32, test_batch_size = 200)
model2 <- SICNN_Net(problem_type = "binary classification", sizes = c(p, 5, 5, 1), input_skip = TRUE, device = "cpu")
train_SICNN(epochs=2000, restarts=1, SICNN=model2, lr=0.01, train_dl=loaders2$train_loader, n_train=800,
           epsilon_1=1, epsilon_T=1e-4, steps_T=1600, sic_threshold=0.5, penalty=log(800))

pdf("paper/figures/tut2_global.pdf", width=8, height=6)
plot(model2, type="global", vertex_size=8, edge_width=0.4, label_size=0.3)
dev.off()
cat("Saved paper/figures/tut2_global.pdf\n")

# --- Figure 3: Tutorial 3 (Gallstone) ---
# Data is in data/Gallstone_Dataset.rda
if (file.exists("data/Gallstone_Dataset.rda")) {
  load("data/Gallstone_Dataset.rda")
  df3 <- Gallstone_Dataset
} else {
  # Fallback: dummy data if not found
  cat("Gallstone_Dataset.rda not found. Check path.\n")
}

# Assume Gallstone_Dataset exists and target is column 'target' or last
target_col <- ncol(df3)
features3 <- df3[,-target_col]
for(i in seq_len(ncol(features3))) {
  r <- range(features3[[i]], na.rm=TRUE)
  if (r[2] != r[1]) features3[[i]] <- (features3[[i]] - r[1]) / (r[2] - r[1])
}
df3_norm <- cbind(features3, df3[[target_col]])
colnames(df3_norm)[ncol(df3_norm)] <- "target"

n3 <- nrow(df3_norm)
n3_train <- floor(n3 * 0.8)
loaders3 <- get_dataloaders(df3_norm, train_proportion = 0.8, train_batch_size = 32, test_batch_size = n3 - n3_train)
model3 <- SICNN_Net(problem_type = "binary classification", sizes = c(ncol(features3), 3, 3, 1), input_skip = TRUE, device = "cpu")
train_SICNN(epochs=1500, restarts=1, SICNN=model3, lr=0.01, train_dl=loaders3$train_loader, n_train=n3_train,
           epsilon_1=1, epsilon_T=1e-4, steps_T=1200, sic_threshold=0.5, penalty=log(n3_train))

pdf("paper/figures/tut3_global.pdf", width=8, height=6)
plot(model3, type="global", vertex_size=10, edge_width=0.4, label_size=0.3)
dev.off()
cat("Saved paper/figures/tut3_global.pdf\n")
