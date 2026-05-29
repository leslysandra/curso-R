# =============================================================================
#  12_REGRESSION_EVALUATION.R
#  Modelos de Predicción Analítica
# =============================================================================
#
#  "A model is only as trustworthy as its evaluation."
#
#  Training a model is step 4 of 6. Evaluating it correctly — on the
#  right data, with the right metrics — is what separates a usable
#  prediction system from one that gives false confidence.
#
#  PACKAGES NEEDED:
#    install.packages(c("tidyverse", "broom", "yardstick",
#                       "tidymodels", "MASS", "ggfortify"))
#
#  DATASETS USED:
#    MASS::Boston (medv), ggplot2::diamonds (price)
#
#  SECTIONS:
#    1.  The golden rule: evaluate on data the model has NOT seen
#    2.  Regression metrics — formulas + intuition
#    3.  Computing metrics in R — three approaches
#    4.  The residual plot toolkit (6 diagnostic charts)
#    5.  Cross-validation — K-fold and Leave-One-Out
#    6.  Comparing multiple models systematically
#    7.  Diagnosing specific failure modes
#    8.  Prediction intervals — honest uncertainty reporting
#    9.  Learning curves — detecting over- and under-fitting
#    10. Complete evaluation report for one model
# =============================================================================

library(tidyverse)
library(broom)
library(yardstick)  # metric functions (rmse, mae, rsq, mape, …)
library(MASS)       # Boston, stepAIC
library(ggfortify)  # autoplot for lm

data(Boston, package = "MASS")
set.seed(2024)

# Train / test split (used throughout this file):
idx       <- sample(nrow(Boston), 0.8 * nrow(Boston))
train_bos <- Boston[ idx, ]
test_bos  <- Boston[-idx, ]

model_ols <- lm(medv ~ ., data = train_bos)
test_bos  <- test_bos |>
  mutate(predicted = predict(model_ols, newdata = test_bos),
         residual  = medv - predicted)


# =============================================================================
# SECTION 1 — THE GOLDEN RULE
# =============================================================================
#
#  NEVER evaluate a model on the data it was trained on.
#
#  Training metrics measure how well the model memorised the training set.
#  Test metrics measure how well it GENERALISES to new observations.
#  Only test metrics tell you whether your model is actually useful.
#
#  HIERARCHY OF TRUSTWORTHINESS:
#   Least trustworthy:  Training set metrics
#   More trustworthy:   Validation set (single hold-out)
#   More trustworthy:   K-fold cross-validation
#   Most trustworthy:   Test set kept sealed until the very end
#
#  COMMON MISTAKES:
#   ✗ Feature engineering using the entire dataset (leakage)
#   ✗ Tuning hyperparameters on the test set (leakage)
#   ✗ Reporting R² on training data as model quality
#   ✗ Using the test set more than once

# Illustrate optimistic bias of training metrics:
train_preds <- predict(model_ols, newdata = train_bos)

rmse_train <- sqrt(mean((train_bos$medv - train_preds)^2))
rmse_test  <- sqrt(mean((test_bos$medv  - test_bos$predicted)^2))

cat(sprintf("Training RMSE : %.3f\n", rmse_train))
cat(sprintf("Test RMSE     : %.3f  ← the number that matters\n", rmse_test))
cat(sprintf("Optimism bias : %.3f  (%.1f%%)\n",
            rmse_test - rmse_train,
            (rmse_test - rmse_train) / rmse_train * 100))


# =============================================================================
# SECTION 2 — REGRESSION METRICS: FORMULAS + INTUITION
# =============================================================================
#
#  Let n = number of test observations
#      yᵢ = actual value
#      ŷᵢ = predicted value
#      ȳ  = mean of actual values (training mean)

# ── MAE — Mean Absolute Error ─────────────────────────────────────────────────
#
#  MAE = (1/n) Σ |yᵢ − ŷᵢ|
#
#  Units:      same as y
#  Robust to:  outliers (uses absolute value, not squared error)
#  Interpret:  "on average, predictions are off by MAE units"
#  Range:      [0, ∞)  — lower is better

# ── RMSE — Root Mean Squared Error ────────────────────────────────────────────
#
#  RMSE = sqrt( (1/n) Σ (yᵢ − ŷᵢ)² )
#
#  Units:      same as y
#  Sensitive:  penalises LARGE errors more than MAE (squared term)
#  Interpret:  "typical prediction error, with heavy penalty on big misses"
#  Range:      [0, ∞)  — lower is better
#  Rule:       RMSE ≥ MAE always; RMSE >> MAE signals influential outliers

# ── MSE — Mean Squared Error ──────────────────────────────────────────────────
#
#  MSE = (1/n) Σ (yᵢ − ŷᵢ)²  = RMSE²
#
#  Units:      y² (less interpretable — prefer RMSE for reporting)
#  Use:        loss function for OLS optimisation; gradient calculations
#  Note:       glmnet, caret, and tidymodels all optimise MSE internally

# ── R² — Coefficient of Determination ────────────────────────────────────────
#
#  R² = 1 − RSS/TSS  =  1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)²
#
#  Proportion of variance in y explained by the model.
#  Range:      (−∞, 1]  —  1 = perfect, 0 = no better than predicting mean
#  Note:       CAN be negative on test data (if predictions are terrible)
#  DO NOT:     compare R² across different datasets or target variables
#  DO NOT:     use training R² as the sole model quality indicator

# ── Adjusted R² ───────────────────────────────────────────────────────────────
#
#  Adj. R² = 1 − (1−R²)(n−1)/(n−p−1)
#
#  Penalises R² for each predictor added.
#  Use only for TRAINING set comparisons of OLS models.
#  Not meaningful on the test set.

# ── MAPE — Mean Absolute Percentage Error ─────────────────────────────────────
#
#  MAPE = (100/n) Σ |yᵢ − ŷᵢ| / |yᵢ|
#
#  Relative (unit-free) — useful for comparing across datasets with
#  different scales, or for business reporting ("off by X%").
#  Problem: undefined or infinite when yᵢ = 0; biased for small yᵢ.
#  MAPE < 10% → excellent;  10–20% → good;  20–50% → acceptable;
#  > 50% → poor (in most business contexts).

# ── MASE — Mean Absolute Scaled Error ────────────────────────────────────────
#
#  Scales MAE by the MAE of a naïve forecast.
#  MASE < 1 → better than the naïve model. Preferred over MAPE.


# =============================================================================
# SECTION 3 — COMPUTING METRICS IN R: THREE APPROACHES
# =============================================================================

actual    <- test_bos$medv
predicted <- test_bos$predicted

# ── Approach 1: Base R (manual formulas) ──────────────────────────────────────
mae  <- mean(abs(actual - predicted))
mse  <- mean((actual - predicted)^2)
rmse <- sqrt(mse)
mape <- mean(abs((actual - predicted) / actual)) * 100
ss_res <- sum((actual - predicted)^2)
ss_tot <- sum((actual - mean(train_bos$medv))^2)
r2   <- 1 - ss_res / ss_tot

cat(sprintf("MAE  : %.3f\n", mae))
cat(sprintf("RMSE : %.3f\n", rmse))
cat(sprintf("MSE  : %.3f\n", mse))
cat(sprintf("MAPE : %.1f%%\n", mape))
cat(sprintf("R²   : %.3f\n", r2))

# ── Approach 2: yardstick (tidyverse-native, recommended) ─────────────────────
results_df <- tibble(
  truth    = actual,
  estimate = predicted
)

metrics(results_df, truth, estimate)    # MAE, RMSE, R² in one call

mae(results_df,  truth, estimate)
rmse(results_df, truth, estimate)
rsq(results_df,  truth, estimate)
mape(results_df, truth, estimate)

# Build a custom metric set:
my_metrics <- metric_set(rmse, mae, rsq, mape)
my_metrics(results_df, truth = truth, estimate = estimate)

# ── Approach 3: A reusable evaluation function ─────────────────────────────────
eval_regression <- function(actual, predicted,
                             model_name = "Model",
                             train_mean = NULL) {
  ss_res  <- sum((actual - predicted)^2)
  ss_tot  <- if (!is.null(train_mean)) {
               sum((actual - train_mean)^2)
             } else {
               sum((actual - mean(actual))^2)
             }
  tibble(
    Model = model_name,
    N     = length(actual),
    MAE   = round(mean(abs(actual - predicted)), 3),
    RMSE  = round(sqrt(mean((actual - predicted)^2)), 3),
    MAPE  = round(mean(abs((actual - predicted)/actual)) * 100, 2),
    R2    = round(1 - ss_res / ss_tot, 4)
  )
}

eval_regression(actual, predicted, "OLS — Boston", mean(train_bos$medv))


# =============================================================================
# SECTION 4 — THE RESIDUAL PLOT TOOLKIT (6 diagnostic charts)
# =============================================================================
#
#  Metrics summarise performance in a single number.
#  Residual plots reveal WHERE and HOW the model fails.
#  Always examine BOTH.

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
#  Ideal: points lie on the 45° red diagonal.
#  Problem — systematic curve: non-linearity not captured
#  Problem — funnel: heteroscedasticity; errors grow with prediction magnitude

p1 <- ggplot(test_bos, aes(x = predicted, y = medv)) +
  geom_point(alpha = 0.6, color = "steelblue", size = 2) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = FALSE, linewidth = 0.8) +
  labs(title = "1. Actual vs Predicted",
       x = "Predicted medv", y = "Actual medv") +
  theme_minimal()

# ── Plot 2: Residuals vs Predicted (Fitted) ───────────────────────────────────
#  Ideal: random scatter around 0, flat LOESS line, constant spread.
#  Problem — U-shape or curve:  non-linearity (try transforming predictors)
#  Problem — funnel shape:      heteroscedasticity (try log(y))

p2 <- ggplot(test_bos, aes(x = predicted, y = residual)) +
  geom_point(alpha = 0.6, color = "steelblue", size = 2) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 0.8) +
  geom_smooth(method = "loess", color = "orange", se = FALSE, linewidth = 0.8) +
  labs(title = "2. Residuals vs Predicted",
       x = "Predicted", y = "Residual (actual − predicted)") +
  theme_minimal()

# ── Plot 3: Histogram of Residuals ────────────────────────────────────────────
#  Ideal: symmetric bell curve centred at 0.
#  Problem — skewed:  consider log-transforming the target
#  Problem — bimodal: the model is missing a structural group

p3 <- ggplot(test_bos, aes(x = residual)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", linewidth = 0.8) +
  labs(title = "3. Histogram of Residuals",
       x = "Residual", y = "Count") +
  theme_minimal()

# ── Plot 4: Q-Q Plot of Residuals ─────────────────────────────────────────────
#  Ideal: points follow the diagonal line exactly.
#  Heavy tails → more extreme errors than expected from a normal distribution.

p4 <- ggplot(test_bos, aes(sample = residual)) +
  stat_qq(color = "steelblue", alpha = 0.7, size = 2) +
  stat_qq_line(color = "red", linetype = "dashed", linewidth = 0.8) +
  labs(title = "4. Q-Q Plot of Residuals",
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

# ── Plot 5: Residuals vs Each Predictor ───────────────────────────────────────
#  Ideal: no systematic pattern (flat LOESS) for each predictor.
#  A pattern with predictor X means X has a non-linear relationship with y
#  that the model hasn't captured.

p5 <- test_bos |>
  select(lstat, rm, dis, residual) |>
  pivot_longer(-residual, names_to = "predictor", values_to = "value") |>
  ggplot(aes(x = value, y = residual)) +
  geom_point(alpha = 0.4, color = "steelblue", size = 1.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = FALSE, linewidth = 0.8) +
  facet_wrap(~ predictor, scales = "free_x") +
  labs(title = "5. Residuals vs Selected Predictors",
       x = "Predictor value", y = "Residual") +
  theme_minimal()

# ── Plot 6: Sorted Actual vs Predicted ────────────────────────────────────────
#  Sorts observations by actual value and plots both actual and predicted.
#  Reveals systematic over- or under-prediction in specific value ranges.

p6 <- test_bos |>
  arrange(medv) |>
  mutate(obs = row_number()) |>
  ggplot(aes(x = obs)) +
  geom_line(aes(y = medv,      color = "Actual"),    linewidth = 0.8) +
  geom_line(aes(y = predicted, color = "Predicted"), linewidth = 0.8,
            linetype = "dashed") +
  scale_color_manual(values = c("Actual" = "grey30", "Predicted" = "steelblue")) +
  labs(title = "6. Sorted Actual vs Predicted",
       x = "Observations (sorted by actual value)",
       y = "medv", color = NULL) +
  theme_minimal()

# Print all 6 plots (in practice use patchwork):
library(patchwork)
(p1 | p2) / (p3 | p4) / (p5 | p6)


# =============================================================================
# SECTION 5 — CROSS-VALIDATION: K-FOLD AND LEAVE-ONE-OUT
# =============================================================================
#
#  A single train/test split can give noisy estimates — different splits
#  produce different results. Cross-validation gives a more stable
#  estimate by averaging over K different train/test splits.
#
#  K-FOLD CV ALGORITHM:
#   1. Split training data into K equal folds.
#   2. For i = 1 to K:
#      a. Use fold i as the validation set.
#      b. Train on the remaining K−1 folds.
#      c. Compute metric on fold i.
#   3. Final metric = average of K fold metrics.
#
#  K = 5 or 10 is the standard. K = n is Leave-One-Out CV (LOOCV).
#  Larger K → less bias, more variance; smaller K → more bias, less variance.

# Manual K-fold cross-validation:
k_fold_cv <- function(data, formula, k = 10, seed = 42) {
  set.seed(seed)
  n       <- nrow(data)
  folds   <- sample(rep(1:k, length.out = n))
  results <- vector("list", k)

  for (i in 1:k) {
    train_fold <- data[folds != i, ]
    val_fold   <- data[folds == i, ]

    model  <- lm(formula, data = train_fold)
    preds  <- predict(model, newdata = val_fold)
    actual <- val_fold[[as.character(formula[[2]])]]

    results[[i]] <- tibble(
      fold = i,
      rmse = sqrt(mean((actual - preds)^2)),
      mae  = mean(abs(actual - preds)),
      r2   = 1 - sum((actual-preds)^2) / sum((actual-mean(train_fold[[as.character(formula[[2]])]])^2))
    )
  }
  bind_rows(results)
}

cv_results <- k_fold_cv(Boston, medv ~ ., k = 10)
cv_results

# Summary:
cv_results |>
  summarise(
    across(c(rmse, mae, r2), list(mean = mean, sd = sd))
  ) |>
  mutate(across(everything(), \(x) round(x, 3)))

# Visualise fold-to-fold variability:
cv_results |>
  pivot_longer(-fold, names_to = "metric", values_to = "value") |>
  ggplot(aes(x = factor(fold), y = value, fill = metric)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ metric, scales = "free_y") +
  labs(title = "10-Fold CV — Metric by Fold",
       x = "Fold", y = "Value") +
  theme_minimal()

# ── Using tidymodels for clean CV ────────────────────────────────────────────
library(tidymodels)

boston_tbl <- as_tibble(Boston)
bos_split  <- initial_split(boston_tbl, prop = 0.8, strata = medv)
bos_train  <- training(bos_split)
bos_test   <- testing(bos_split)
bos_folds  <- vfold_cv(bos_train, v = 10, strata = medv)

ols_spec <- linear_reg() |> set_engine("lm")
ols_wf   <- workflow() |>
  add_formula(medv ~ .) |>
  add_model(ols_spec)

cv_tidy <- fit_resamples(
  ols_wf,
  resamples = bos_folds,
  metrics   = metric_set(rmse, mae, rsq)
)

collect_metrics(cv_tidy)   # mean + std_err across 10 folds


# =============================================================================
# SECTION 6 — COMPARING MULTIPLE MODELS SYSTEMATICALLY
# =============================================================================

# Fit four models of increasing complexity:
f1 <- medv ~ lstat
f2 <- medv ~ lstat + rm
f3 <- medv ~ lstat + rm + crim + nox + dis
f4 <- medv ~ .

model_list <- list(
  "lstat only"    = lm(f1, data = train_bos),
  "lstat + rm"    = lm(f2, data = train_bos),
  "5 predictors"  = lm(f3, data = train_bos),
  "All predictors"= lm(f4, data = train_bos)
)

# Evaluate each on the test set:
comparison <- map_dfr(names(model_list), function(nm) {
  preds  <- predict(model_list[[nm]], newdata = test_bos)
  actual <- test_bos$medv
  eval_regression(actual, preds, nm, mean(train_bos$medv))
})

comparison |> arrange(RMSE)

# Visual comparison:
comparison |>
  pivot_longer(c(MAE, RMSE, R2), names_to = "metric", values_to = "value") |>
  ggplot(aes(x = reorder(Model, value), y = value, fill = Model)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ metric, scales = "free") +
  coord_flip() +
  labs(title = "Model Comparison — Test Set Metrics",
       x = NULL, y = "Value") +
  theme_minimal()


# =============================================================================
# SECTION 7 — DIAGNOSING SPECIFIC FAILURE MODES
# =============================================================================

# ── Failure Mode 1: Heteroscedasticity ────────────────────────────────────────
#  Residual spread grows with the fitted value.
#  Diagnosis: funnel shape in Residuals vs Predicted.
#  Fix: log-transform the target.

model_log <- lm(log(medv) ~ ., data = train_bos)
test_bos  <- test_bos |>
  mutate(
    pred_log    = exp(predict(model_log, newdata = test_bos)),
    resid_log   = medv - pred_log
  )

ggplot(test_bos, aes(x = pred_log, y = resid_log)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = FALSE) +
  labs(title = "Residuals vs Predicted — log(medv) model",
       subtitle = "Compare spread to the OLS model") +
  theme_minimal()

eval_regression(test_bos$medv, test_bos$pred_log,
                "OLS — log(medv)", mean(train_bos$medv))

# ── Failure Mode 2: Structural non-linearity ───────────────────────────────────
#  Residuals form a systematic curve vs a predictor.
#  Diagnosis: curved LOESS in Residuals vs predictor X.
#  Fix: add polynomial or log-transform X.

test_bos <- test_bos |>
  mutate(
    resid_ols = medv - predicted   # original OLS residual
  )

ggplot(test_bos, aes(x = lstat, y = resid_ols)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  labs(title = "Residuals vs lstat — non-linear pattern suggests log(lstat)",
       x = "lstat", y = "Residual") +
  theme_minimal()

# Fix: log-transform lstat
model_log_lstat <- lm(medv ~ log(lstat) + rm + crim + nox +
                              dis + rad + tax + ptratio + b + zn +
                              indus + chas + age, data = train_bos)
test_bos <- test_bos |>
  mutate(pred_log_lstat = predict(model_log_lstat, newdata = test_bos))

eval_regression(test_bos$medv, test_bos$pred_log_lstat,
                "OLS — log(lstat)", mean(train_bos$medv))

# ── Failure Mode 3: Large errors in a specific range ──────────────────────────
#  Some models systematically over- or under-predict high/low values.
#  Diagnosis: Sorted Actual vs Predicted (Plot 6).

test_bos |>
  mutate(error_pct = abs(residual) / medv * 100,
         medv_band = cut(medv, breaks = quantile(medv, c(0,.25,.5,.75,1)),
                         include.lowest = TRUE)) |>
  group_by(medv_band) |>
  summarise(mean_error_pct = round(mean(error_pct), 1),
            n = n())


# =============================================================================
# SECTION 8 — PREDICTION INTERVALS: HONEST UNCERTAINTY REPORTING
# =============================================================================
#
#  A point prediction alone is not enough.
#  A prediction interval gives the range where the true value
#  is expected to fall with a specified probability (e.g. 95%).
#
#  Coverage = proportion of test observations that fall WITHIN the interval.
#  A well-calibrated 95% PI should contain ≈ 95% of test observations.

preds_with_pi <- predict(model_ols,
                          newdata = test_bos,
                          interval = "prediction",
                          level = 0.95) |>
  as.data.frame() |>
  rename(pred = fit, pi_low = lwr, pi_high = upr)

test_bos_pi <- bind_cols(test_bos, preds_with_pi) |>
  mutate(in_interval = medv >= pi_low & medv <= pi_high)

# Empirical coverage:
cat(sprintf("PI coverage: %.1f%% (target: 95%%)\n",
            mean(test_bos_pi$in_interval) * 100))

# PI width — narrower = more precise model:
mean(preds_with_pi$pi_high - preds_with_pi$pi_low)

# Visualise 30 random predictions with their PIs:
test_bos_pi |>
  slice_sample(n = 30) |>
  arrange(pred) |>
  mutate(obs = row_number()) |>
  ggplot(aes(x = obs)) +
  geom_errorbar(aes(ymin = pi_low, ymax = pi_high, color = in_interval),
                width = 0.3, alpha = 0.8) +
  geom_point(aes(y = medv), size = 2, color = "black") +
  geom_point(aes(y = pred), size = 2, shape = 4, color = "red") +
  scale_color_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato"),
                     labels = c("Outside", "Inside")) +
  labs(title = "95% Prediction Intervals — 30 Test Observations",
       subtitle = "Dot = actual | X = predicted | Bar = PI | Red = actual outside PI",
       x = "Observation (sorted by prediction)", y = "medv", color = "In PI") +
  theme_minimal()


# =============================================================================
# SECTION 9 — LEARNING CURVES: DIAGNOSING OVER- AND UNDER-FITTING
# =============================================================================
#
#  A learning curve plots training and validation error against
#  the amount of training data used.
#
#  PATTERNS TO RECOGNISE:
#
#   Underfitting (high bias):
#     Both curves plateau at HIGH error.
#     Adding more data won't help — the model is too simple.
#     Fix: add more features, increase model complexity.
#
#   Overfitting (high variance):
#     Training error is LOW; validation error is HIGH.
#     A large GAP between the two curves.
#     Fix: more training data, regularisation, simpler model.
#
#   Good fit:
#     Both curves converge to a LOW error.

compute_learning_curve <- function(formula, data, target_col,
                                   fractions = seq(0.1, 1, 0.05),
                                   n_reps = 5) {
  n <- nrow(data)
  map_dfr(fractions, function(f) {
    map_dfr(1:n_reps, function(r) {
      set.seed(r * 100)
      idx_train <- sample(n, max(10, floor(f * n * 0.8)))
      idx_test  <- setdiff(1:n, seq_len(floor(0.8 * n)))
      if (length(idx_test) < 5) return(NULL)

      tr <- data[idx_train, ]
      te <- data[sample(setdiff(1:n, idx_train), min(100, length(setdiff(1:n, idx_train)))), ]

      m    <- lm(formula, data = tr)
      yhat_tr <- predict(m, newdata = tr)
      yhat_te <- predict(m, newdata = te)

      y_tr <- tr[[target_col]]
      y_te <- te[[target_col]]

      tibble(
        fraction   = f,
        n_train    = nrow(tr),
        train_rmse = sqrt(mean((y_tr - yhat_tr)^2)),
        val_rmse   = sqrt(mean((y_te - yhat_te)^2))
      )
    })
  })
}

lc <- compute_learning_curve(medv ~ ., Boston, "medv")

lc_summary <- lc |>
  group_by(n_train) |>
  summarise(
    train_rmse = mean(train_rmse),
    val_rmse   = mean(val_rmse),
    .groups = "drop"
  )

ggplot(lc_summary, aes(x = n_train)) +
  geom_line(aes(y = train_rmse, color = "Training"), linewidth = 1) +
  geom_line(aes(y = val_rmse,   color = "Validation"), linewidth = 1) +
  scale_color_manual(values = c("Training" = "steelblue",
                                 "Validation" = "tomato")) +
  labs(title = "Learning Curve — Boston OLS Model",
       subtitle = "Gap between curves → overfitting | Both high → underfitting",
       x = "Training set size", y = "RMSE", color = NULL) +
  theme_minimal()


# =============================================================================
# SECTION 10 — COMPLETE EVALUATION REPORT FOR ONE MODEL
# =============================================================================

cat("\n", strrep("=", 60), "\n", sep = "")
cat("  REGRESSION MODEL EVALUATION REPORT\n")
cat("  Dataset: Boston Housing | Target: medv\n")
cat("  Model:   OLS — all predictors\n")
cat(strrep("=", 60), "\n\n", sep = "")

# ── 1. Dataset summary ────────────────────────────────────────────────────────
cat(sprintf("Training observations : %d\n", nrow(train_bos)))
cat(sprintf("Test observations     : %d\n", nrow(test_bos)))
cat(sprintf("Number of predictors  : %d\n", ncol(Boston) - 1))
cat("\n")

# ── 2. Model summary statistics ───────────────────────────────────────────────
gs <- glance(model_ols)
cat(sprintf("Training R²           : %.4f\n", gs$r.squared))
cat(sprintf("Training Adj. R²      : %.4f\n", gs$adj.r.squared))
cat(sprintf("Training RSE          : %.3f\n", gs$sigma))
cat(sprintf("Training AIC          : %.1f\n",  gs$AIC))
cat("\n")

# ── 3. Test set performance ───────────────────────────────────────────────────
test_mae  <- mean(abs(test_bos$medv - test_bos$predicted))
test_rmse <- sqrt(mean((test_bos$medv - test_bos$predicted)^2))
test_mape <- mean(abs((test_bos$medv - test_bos$predicted) / test_bos$medv)) * 100
test_r2   <- 1 - sum((test_bos$medv - test_bos$predicted)^2) /
                  sum((test_bos$medv - mean(train_bos$medv))^2)

cat(sprintf("Test MAE              : %.3f  (avg error in $1,000)\n", test_mae))
cat(sprintf("Test RMSE             : %.3f  (penalises large errors more)\n", test_rmse))
cat(sprintf("Test MAPE             : %.1f%% (avg %% error)\n", test_mape))
cat(sprintf("Test R²               : %.4f (%% variance explained)\n", test_r2))
cat("\n")

# ── 4. Coefficient table ──────────────────────────────────────────────────────
cat("Coefficients (training model):\n")
tidy(model_ols, conf.int = TRUE) |>
  mutate(
    significant = ifelse(p.value < 0.05, "Yes", "No"),
    across(where(is.numeric), \(x) round(x, 3))
  ) |>
  select(term, estimate, std.error, conf.low, conf.high, p.value, significant) |>
  print(n = Inf)

# ── 5. Worst predictions ──────────────────────────────────────────────────────
cat("\n10 Observations with largest absolute errors:\n")
test_bos |>
  mutate(abs_error = abs(residual),
         pct_error = round(abs_error / medv * 100, 1)) |>
  arrange(desc(abs_error)) |>
  select(medv, predicted, residual, abs_error, pct_error) |>
  head(10) |>
  print()

cat("\n", strrep("=", 60), "\n", sep = "")
cat("  END OF REPORT\n")
cat(strrep("=", 60), "\n\n", sep = "")


# =============================================================================
#  METRICS QUICK-REFERENCE CARD
# =============================================================================
#
#  METRIC   FORMULA                  UNITS   ROBUST  INTERPRETATION
#  MAE      mean(|y − ŷ|)           y       Yes     Average absolute error
#  RMSE     sqrt(mean((y−ŷ)²))      y       No      Penalises large errors
#  MSE      mean((y−ŷ)²)            y²      No      OLS loss function
#  MAPE     mean(|y−ŷ|/|y|)×100    %       No      Relative error %
#  R²       1 − RSS/TSS             —       No      Variance explained
#
#  RESIDUAL PLOTS:
#    Actual vs Predicted        → systematic bias detection
#    Residuals vs Predicted     → non-linearity / heteroscedasticity
#    Histogram of residuals     → normality of errors
#    Q-Q plot                   → tail behaviour
#    Residuals vs predictor     → missing non-linear terms
#    Sorted actual vs predicted → range-specific failures
#
#  VALIDATION STRATEGIES:
#    Hold-out (80/20 split)     → fast, noisy for small datasets
#    K-fold CV (K=10)           → standard, stable
#    LOOCV                      → unbiased but slow (use for small n)
#    Nested CV                  → needed when tuning hyperparameters
#
# =============================================================================
#  END OF 12_REGRESSION_EVALUATION.R
# =============================================================================
