library(devtools)
load_all(".")
set.seed(42)
torch::torch_manual_seed(42)
p <- 15
model <- SICNN_Net("regression", sizes = c(p, 5, 5, 1), input_skip = TRUE, device="cpu")
model$compute_paths_input_skip(epsilon=1e-5, threshold=0.5)

cat("\nLayer 1 dims:\n")
l1 <- model$layers$children[[1]]
alp1 <- as.matrix(l1$alpha_active_path$cpu())
print(dim(alp1))

cat("\nLayer 2 dims:\n")
l2 <- model$layers$children[[2]]
alp2 <- as.matrix(l2$alpha_active_path$cpu())
print(dim(alp2))
