# DPMM baseline on lambda1 (Kottas, 2006): Dirichlet process mixture of Beta
# kernels fitted by MCMC (R package dirichletprocess), for Table 1 and Figures
# 3, 4 and 12. Run from the repository root:
#
#   Rscript experiments/dpmm_lambda1.R
#
# Requires install.packages(c("dirichletprocess", "reticulate")); reticulate
# needs a Python with NumPy to write the results. Writes logs/lambda1/dpmm/ in
# the same format as experiments/run.py. Note that the intensity is evaluated
# on seq(0.1, 49.9, length.out = 160), while the other methods use
# linspace(1e-6, 50 - 1e-6, 160).

library(dirichletprocess)
library(reticulate)

B <- 1000
num_seeds <- 50
grid_size <- 160

# Pre-allocate array: [Seeds, B, Grid_Points]
dpmm_lambda1 <- array(NA, dim = c(num_seeds, B, grid_size))
dpmm_times_lambda1 <- numeric(num_seeds)
dpmm_lambda1_ll <- numeric(num_seeds)

# ---- Loop over seeds ----
for (s in 0:(num_seeds-1)) {
  cat("Processing seed:", s, "\n")
  set.seed(42 + s)  # seed s uses the stream seeded with 42 + s, as in experiments/run.py

  data_path <- paste0("data/lambda1/lambda1_", s, ".csv")
  event_data <- read.csv(data_path)
  x_coords <- event_data$x1
  n <- length(x_coords)
  
  # Start timer
  t_start <- Sys.time()
  
  # Fit DP
  dp <- DirichletProcessBeta(
    y = x_coords,
    maxY = 50,
    g0Priors = c(2, 4),
    alphaPrior = c(2, 8),
    mhStep = c(1, 1),
    hyperPriorParameters = c(1, 0.125),
    verbose = FALSE
  )
  
  dp <- Fit(dp, its = 2000, updatePrior = TRUE, progressBar = TRUE)
  
  # Compute Intensities
  x_grid <- seq(1e-1, 50 - 1e-1, length.out = grid_size)
  
  densities <- t(replicate(B, PosteriorFunction(dp)(x_grid)))
  gamma_draws <- rgamma(B, shape = n, rate = 1)
  dpmm_lambda1[s + 1, , ] <- densities * gamma_draws
  
  # End timer
  t_end <- Sys.time()
  
  # Calculate integral
  dx <- x_grid[2] - x_grid[1]
  integral_terms <- rowSums(dpmm_lambda1[s + 1, , ]) * dx

  # -----------------------------
  # Test log likelihood
  # -----------------------------
  other_seeds <- setdiff(0:(num_seeds - 1), s)
  sum_ll <- 0

for (other_seed in other_seeds) {
    cat("Testing on seed:", other_seed, "\n")

    # Load test data
    test_path <- paste0("data/lambda1/lambda1_", other_seed, ".csv")
    test_data <- read.csv(test_path)
    x_test <- test_data$x1

    # Evaluate posterior density at test points
    test_densities <- t(replicate(B, PosteriorFunction(dp)(x_test)))
    log_intensities <- log(pmax(test_densities, 1e-300)) + log(gamma_draws)
    log_likelihoods <- (-integral_terms + rowSums(log_intensities))

    # log(mean(exp(log_likelihoods)))
    max_ll <- max(log_likelihoods)
    loglik <- (max_ll + log(mean(exp(log_likelihoods - max_ll))))

    cat("  Test log likelihood:", loglik, "\n")
    sum_ll <- sum_ll + loglik
  }

  # Average over all other datasets
  dpmm_lambda1_ll[s + 1] <- (sum_ll / length(other_seeds))

  cat("Average test log likelihood:", dpmm_lambda1_ll[s + 1], "\n")

  dpmm_times_lambda1[s + 1] <- as.numeric(difftime(t_end, t_start, units = "secs"))
}

# ---- Save as .npy files ----

out_dir <- "logs/lambda1/dpmm"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

np <- import("numpy")
np$save(file.path(out_dir, "intensities.npy"), dpmm_lambda1)
np$save(file.path(out_dir, "runtime.npy"), dpmm_times_lambda1)
np$save(file.path(out_dir, "test_loglik.npy"), dpmm_lambda1_ll)
np$save(file.path(out_dir, "seeds.npy"), np$arange(num_seeds))

cat("Files saved successfully to", out_dir, "\n")