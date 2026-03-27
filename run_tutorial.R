# Wrapper to load local SICNN package and run the tutorial
library(devtools)
load_all(".")

# Run tutorial parts but print results
source("rj_experiments/Tutorial_1_linear_SIC.R", echo=TRUE)

# Explicit check for coef with uncertainty
cat("\n--- Testing coef with uncertainty ---\n")
res <- coef(
  model_input_skip,
  dataset = train_loader,
  inds = c(1),
  uncertainty = TRUE,
  fisher_dataloader = test_loader
)
print(res)

cat("\n--- Completed tutorial and extra checks ---\n")
