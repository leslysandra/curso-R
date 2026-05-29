# =============================================================================
#  10_LINEAR_REGRESSION.R
#  Modelos de Predicción Analítica
# =============================================================================
#
#  Linear regression is the foundation of predictive modelling.
#  Every advanced technique — regularisation, GLMs, neural networks —
#  builds on the concepts introduced here. Master this file first.
#
#  PACKAGES NEEDED:
#    install.packages(c("tidyverse", "broom", "car", "lmtest",
#                       "ggfortify", "MASS", "GGally"))
#
#  DATASETS USED:
#    mtcars (built-in), MASS::Boston
#
#  SECTIONS:
#    1.  The linear model — mathematical foundation
#    2.  Simple linear regression   (one predictor)
#    3.  Multiple linear regression (many predictors)
#    4.  Reading the summary() output — every number explained
#    5.  The five OLS assumptions and how to test them
#    6.  Residual diagnostics — the four base R plots
#    7.  Influential observations — leverage and Cook's distance
#    8.  Model selection — forward, backward, stepwise AIC
#    9.  Polynomial regression — capturing non-linearity
#    10. Presenting results cleanly with broom
#    11. Prediction — point estimates and confidence intervals
#    12. Complete modelling workflow end-to-end
# =============================================================================

library(tidyverse)
library(broom)      # tidy(), glance(), augment()
library(car)        # vif(), outlierTest()
library(lmtest)     # bptest(), dwtest()
library(ggfortify)  # autoplot() for lm objects
library(MASS)       # Boston dataset, stepAIC()
library(GGally)     # ggpairs()

data(mtcars)
data(Boston, package = "MASS")


# =============================================================================
# SECTION 1 — THE LINEAR MODEL: MATHEMATICAL FOUNDATION
# =============================================================================
#
#  Simple:   y  =  β₀  +  β₁x₁  +  ε
#  Multiple: y  =  β₀  +  β₁x₁  +  β₂x₂  + … +  βₚxₚ  +  ε
#
#  WHERE:
#   y       = target (dependent) variable — what we predict
#   x₁…xₚ  = predictors (independent variables / features)
#   β₀      = intercept — predicted y when all x = 0
#   β₁…βₚ  = slopes — change in y per one-unit increase in xⱼ,
#             holding all other predictors constant (ceteris paribus)
#   ε       = error term — assumed ~ N(0, σ²)
#
#  OLS (Ordinary Least Squares) ESTIMATION:
#   Finds the β values that minimise  RSS = Σ(yᵢ − ŷᵢ)²
#   where ŷᵢ = β₀ + β₁x₁ᵢ + … is the model's prediction.
#
#  KEY TERMINOLOGY:
#   Residual     = observed − predicted = yᵢ − ŷᵢ
#   Fitted value = the model's prediction = ŷᵢ
#   R²           = proportion of variance in y explained by the model
#   Adj. R²      = R² penalised for number of predictors
#   RSE          = Residual Standard Error — average residual size (in y units)


# =============================================================================
# SECTION 2 — SIMPLE LINEAR REGRESSION (one predictor)
# =============================================================================

# Question: does a car's WEIGHT predict its FUEL EFFICIENCY?

# Step 1: Visualise the relationship first — ALWAYS
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 3, color = "steelblue", alpha = 0.8) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Weight vs Fuel Efficiency — mtcars",
       x = "Weight (1,000 lbs)", y = "Miles per Gallon") +
  theme_minimal()

# Step 2: Fit the model
#  Syntax:  lm(target ~ predictor, data = dataset)
model_simple <- lm(mpg ~ wt, data = mtcars)

# Step 3: Inspect the model
summary(model_simple)

# Step 4: Extract individual components
coef(model_simple)          # β₀ (Intercept) and β₁ (wt)
residuals(model_simple)     # ε̂ = y − ŷ  for each observation
fitted(model_simple)        # ŷ (predictions on training data)
sigma(model_simple)         # Residual Standard Error (RSE)
confint(model_simple)       # 95% confidence intervals for coefficients

# INTERPRETATION:
#  Intercept (β₀ ≈ 37.3): predicted MPG when wt = 0 (theoretical baseline)
#  wt slope  (β₁ ≈ −5.3): each extra 1,000 lbs REDUCES MPG by 5.3 units
#  p-value for wt < 0.001: weight is a highly significant predictor
#  R² = 0.75: weight alone explains 75% of the variance in MPG


# =============================================================================
# SECTION 3 — MULTIPLE LINEAR REGRESSION (many predictors)
# =============================================================================

# Explore pairwise relationships first
mtcars |>
  select(mpg, wt, hp, disp, drat) |>
  ggpairs(title = "Predictor Relationships — mtcars")

# Fit a multiple regression model  (separate predictors with +)
model_multi <- lm(mpg ~ wt + hp + factor(cyl), data = mtcars)
summary(model_multi)

# Formula notation cheat sheet:
#  mpg ~ .           use ALL other columns as predictors
#  mpg ~ . - disp    all columns EXCEPT disp
#  mpg ~ wt + hp     only wt and hp
#  mpg ~ wt * hp     wt + hp + their interaction (wt:hp)
#  mpg ~ wt:hp       ONLY the interaction term
#  mpg ~ poly(wt,2)  wt + wt² (polynomial — see Section 9)

# Compare models with glance() from broom
model_1 <- lm(mpg ~ wt,                    data = mtcars)
model_2 <- lm(mpg ~ wt + hp,               data = mtcars)
model_3 <- lm(mpg ~ wt + hp + factor(cyl), data = mtcars)

bind_rows(
  glance(model_1) |> mutate(model = "wt only"),
  glance(model_2) |> mutate(model = "wt + hp"),
  glance(model_3) |> mutate(model = "wt + hp + cyl")
) |>
  select(model, r.squared, adj.r.squared, sigma, AIC, BIC) |>
  arrange(AIC)

# Formal model comparison with an F-test (ANOVA):
anova(model_1, model_2)   # does adding hp significantly improve fit?
anova(model_2, model_3)   # does adding cyl significantly improve fit?
# p < 0.05 → the more complex model fits significantly better


# =============================================================================
# SECTION 4 — READING THE summary() OUTPUT — EVERY NUMBER EXPLAINED
# =============================================================================

summary(model_multi)

# ── OUTPUT ANATOMY ────────────────────────────────────────────────────────────
#
# Call:
#   The exact formula you used — confirms you fitted what you intended.
#
# Residuals: Min / 1Q / Median / 3Q / Max
#   Five-number summary of residuals.
#   Ideal: Median ≈ 0 (unbiased), Min ≈ −Max (symmetric).
#
# Coefficients table:
#   Estimate   → β̂ⱼ — the fitted coefficient
#   Std. Error → uncertainty in the estimate
#   t value    → Estimate / Std.Error (distance from zero in SE units)
#   Pr(>|t|)   → p-value: prob. of |t| this large if true β = 0
#   Signif: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1
#
# Residual standard error (RSE):
#   Average prediction error in the UNITS of y.
#   RSE = sqrt(RSS / (n − p − 1))  where p = number of predictors.
#   "On average, predictions are off by ±RSE mpg."
#
# R-squared:
#   Proportion of variance in y explained by the model: R² ∈ [0, 1].
#   NEVER compare R² across models with different numbers of predictors.
#
# Adjusted R-squared:
#   R² penalised for each additional predictor added.
#   USE THIS when comparing models of different sizes.
#   Adj.R² decreases if a predictor adds no real explanatory power.
#
# F-statistic and its p-value:
#   H₀: all β₁…βₚ = 0 (the model is no better than predicting the mean).
#   p < 0.05 → at least one predictor is significantly related to y.


# =============================================================================
# SECTION 5 — THE FIVE OLS ASSUMPTIONS AND HOW TO TEST THEM
# =============================================================================
#
#  OLS gives BEST LINEAR UNBIASED ESTIMATES only when these hold.
#  Violations affect standard errors, p-values, and CIs — not always β̂.
#
#  A1. LINEARITY         relationship between x and y is linear
#  A2. INDEPENDENCE      residuals are independent of each other
#  A3. HOMOSCEDASTICITY  residuals have constant variance across fitted values
#  A4. NORMALITY         residuals ~ Normal (needed for valid inference)
#  A5. NO MULTICOLLINEARITY  predictors are not highly correlated

model_diag <- lm(mpg ~ wt + hp + disp, data = mtcars)

# A1. LINEARITY
# Residuals vs Fitted plot — look for a FLAT red line centred at 0.
# A curve signals a non-linear relationship.
plot(model_diag, which = 1)
# Remedies: polynomial terms, log-transform predictors, non-linear models.

# A2. INDEPENDENCE
# Durbin-Watson test — detects autocorrelation. Critical for time-series.
dwtest(model_diag)
# DW ≈ 2 → no autocorrelation (good)
# DW < 1.5 or > 2.5 → potential autocorrelation

# A3. HOMOSCEDASTICITY (constant variance)
plot(model_diag, which = 3)   # Scale-Location — red line should be flat
bptest(model_diag)            # Breusch-Pagan formal test
# H₀: constant variance.  p < 0.05 → heteroscedasticity detected.
# Remedies: log-transform target, Weighted Least Squares, robust SEs.

# A4. NORMALITY OF RESIDUALS
plot(model_diag, which = 2)              # Q-Q plot — points on the diagonal
shapiro.test(residuals(model_diag))      # H₀: residuals are normal
hist(residuals(model_diag), breaks = 15,
     main = "Histogram of Residuals",
     col = "steelblue", border = "white")
abline(v = 0, col = "red", lwd = 2, lty = 2)
# Note: with n > 30 mild non-normality has little practical effect.

# A5. NO MULTICOLLINEARITY
vif(model_diag)
# VIF = 1: no correlation | < 5: acceptable | 5–10: moderate | > 10: severe
# Remedies: remove a correlated predictor, create a ratio, use Ridge regression.


# =============================================================================
# SECTION 6 — RESIDUAL DIAGNOSTICS: THE FOUR BASE R PLOTS
# =============================================================================

par(mfrow = c(2, 2))
plot(model_multi)
par(mfrow = c(1, 1))

# PLOT 1 — RESIDUALS vs FITTED
#   Ideal: random scatter around 0, flat red line.
#   Problem — curve:     non-linearity (A1 violated)
#   Problem — funnel:    heteroscedasticity (A3 violated)

# PLOT 2 — NORMAL Q-Q
#   Ideal: points follow the diagonal dashed line.
#   Problem — S-curve:   skewed residuals
#   Problem — heavy tails: influential outliers

# PLOT 3 — SCALE-LOCATION
#   Ideal: horizontal red line, equally spread points.
#   Problem — rising/falling line: variance changes with fitted value (A3)

# PLOT 4 — RESIDUALS vs LEVERAGE
#   Leverage: how unusual are the observation's X values?
#   Cook's Distance: overall influence on all β̂ estimates.
#   Problem: points outside Cook's distance contours (dashed lines)
#     → these observations strongly influence the fitted coefficients.

# ggplot2 version via ggfortify:
autoplot(model_multi, which = 1:4,
         colour = "steelblue",
         smooth.colour = "red") +
  theme_minimal()


# =============================================================================
# SECTION 7 — INFLUENTIAL OBSERVATIONS: LEVERAGE AND COOK'S DISTANCE
# =============================================================================

model_aug <- augment(model_multi, data = mtcars)
# .fitted    = ŷᵢ
# .resid     = yᵢ − ŷᵢ
# .std.resid = standardised residual (in SD units)
# .hat       = leverage hᵢ
# .cooksd    = Cook's distance Dᵢ

# Leverage threshold: hᵢ > 2(p+1)/n
p <- length(coef(model_multi)) - 1
n <- nrow(mtcars)
lev_thresh  <- 2 * (p + 1) / n
cook_thresh <- 4 / n

cat("Leverage threshold:", round(lev_thresh,  3), "\n")
cat("Cook's D threshold:", round(cook_thresh, 3), "\n")

# High-leverage observations:
model_aug |> filter(.hat > lev_thresh) |>
  select(.hat, .cooksd, .std.resid) |> arrange(desc(.hat))

# Influential observations by Cook's D:
model_aug |> filter(.cooksd > cook_thresh) |>
  arrange(desc(.cooksd)) |> select(.hat, .cooksd, .std.resid)

# Visual Cook's distance:
ggplot(model_aug, aes(x = seq_along(.cooksd), y = .cooksd)) +
  geom_col(fill = "steelblue") +
  geom_hline(yintercept = cook_thresh, color = "red", linetype = "dashed") +
  labs(title = "Cook's Distance — Influential Observations",
       x = "Observation Index", y = "Cook's Distance") +
  theme_minimal()

# Formal outlier test (Bonferroni-corrected):
outlierTest(model_multi)

# Compare model with vs without the most influential point:
most_inf <- which.max(model_aug$.cooksd)
model_excl <- lm(mpg ~ wt + hp + factor(cyl), data = mtcars[-most_inf, ])
round(coef(model_multi) - coef(model_excl), 3)   # coefficient shift


# =============================================================================
# SECTION 8 — MODEL SELECTION: STEPWISE AIC
# =============================================================================
#
#  AIC = 2k − 2ln(L)  where k = parameters, L = likelihood.
#  Lower AIC → better model (penalises unnecessary complexity).

model_full <- lm(mpg ~ ., data = mtcars)
model_null <- lm(mpg ~ 1, data = mtcars)

# Backward elimination (start full, remove one at a time):
model_back <- stepAIC(model_full, direction = "backward", trace = FALSE)
summary(model_back)

# Forward selection (start null, add one at a time):
model_fwd  <- stepAIC(model_null,
                       direction = "forward",
                       scope = list(lower = model_null, upper = model_full),
                       trace = FALSE)

# Bidirectional stepwise (most common in practice):
model_step <- stepAIC(model_full, direction = "both", trace = FALSE)

# Compare AIC across all approaches:
AIC(model_null, model_back, model_fwd, model_step, model_full)

# ⚠️  CAVEAT: stepwise inflates Type I error and may overfit.
#     Regularisation (11_regularization.R) is preferred for predictive tasks.


# =============================================================================
# SECTION 9 — POLYNOMIAL REGRESSION: CAPTURING NON-LINEARITY
# =============================================================================

ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 3, color = "steelblue") +
  geom_smooth(method = "lm",                        color = "red",
              linetype = "dashed", se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2),
              color = "darkgreen", se = FALSE) +
  labs(title = "Linear (red) vs Quadratic (green) fit",
       x = "Weight (klb)", y = "MPG") +
  theme_minimal()

model_lin  <- lm(mpg ~ wt,          data = mtcars)
model_quad <- lm(mpg ~ poly(wt, 2), data = mtcars)  # orthogonal polynomials
model_cub  <- lm(mpg ~ poly(wt, 3), data = mtcars)

bind_rows(
  glance(model_lin)  |> mutate(model = "Linear"),
  glance(model_quad) |> mutate(model = "Quadratic"),
  glance(model_cub)  |> mutate(model = "Cubic")
) |> select(model, r.squared, adj.r.squared, AIC)

# Is the quadratic term worth adding?
anova(model_lin,  model_quad)  # p < 0.05 → yes
anova(model_quad, model_cub)   # p > 0.05 → cubic adds nothing

# ⚠️  Higher degree = more training fit but worse generalisation.
#     Always evaluate on held-out test data.


# =============================================================================
# SECTION 10 — PRESENTING RESULTS CLEANLY WITH broom
# =============================================================================

model_final <- lm(mpg ~ wt + hp + factor(cyl), data = mtcars)

# Coefficient table:
tidy(model_final, conf.int = TRUE) |>
  mutate(
    stars     = case_when(
      p.value < 0.001 ~ "***", p.value < 0.01 ~ "**",
      p.value < 0.05  ~ "*",   p.value < 0.1  ~ ".",
      TRUE ~ ""
    ),
    across(where(is.numeric), \(x) round(x, 3))
  ) |>
  select(term, estimate, std.error, conf.low, conf.high, p.value, stars)

# Model-level stats:
glance(model_final) |>
  select(r.squared, adj.r.squared, sigma, statistic, p.value, AIC, BIC, nobs)

# Coefficient plot — estimates + uncertainty at a glance:
tidy(model_final, conf.int = TRUE) |>
  filter(term != "(Intercept)") |>
  ggplot(aes(x = estimate, y = reorder(term, estimate))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 height = 0.2, color = "steelblue", linewidth = 1) +
  geom_point(size = 3, color = "steelblue") +
  labs(title = "Coefficient Plot — 95% Confidence Intervals",
       x = "Estimate (β)", y = NULL) +
  theme_minimal()


# =============================================================================
# SECTION 11 — PREDICTION: POINT ESTIMATES AND CONFIDENCE INTERVALS
# =============================================================================
#
#  Confidence Interval (CI):  uncertainty about the MEAN response for x
#  Prediction Interval (PI):  uncertainty about ONE NEW individual observation
#  PI > CI always — it includes model uncertainty PLUS residual variance ε.

new_cars <- data.frame(wt = c(2.5, 3.5, 4.5),
                       hp = c(100, 150, 200),
                       cyl = factor(c(4, 6, 8)))

predict(model_final, newdata = new_cars)                              # point
predict(model_final, newdata = new_cars, interval = "confidence")    # CI
predict(model_final, newdata = new_cars, interval = "prediction")    # PI

# Visualise both intervals over the wt range:
grid <- data.frame(wt = seq(1.5, 5.5, 0.05), hp = 150, cyl = factor(6))
ci_band <- as.data.frame(predict(model_final, grid, interval = "confidence"))
pi_band <- as.data.frame(predict(model_final, grid, interval = "prediction"))
grid_df <- bind_cols(grid, rename(ci_band, ci_lwr=lwr, ci_upr=upr),
                            rename(pi_band, pi_lwr=lwr, pi_upr=upr, fit2=fit))

ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_ribbon(data = grid_df, aes(y = fit, ymin = pi_lwr, ymax = pi_upr),
              fill = "lightblue", alpha = 0.4) +
  geom_ribbon(data = grid_df, aes(y = fit, ymin = ci_lwr, ymax = ci_upr),
              fill = "steelblue", alpha = 0.5) +
  geom_line(data = grid_df, aes(y = fit), color = "steelblue", linewidth = 1.2) +
  geom_point(size = 2.5, color = "grey30") +
  labs(title = "95% CI (dark) and PI (light) around the fitted line",
       x = "Weight (klb)", y = "MPG") +
  theme_minimal()


# =============================================================================
# SECTION 12 — COMPLETE WORKFLOW: BOSTON HOUSING
# =============================================================================

glimpse(Boston)  # 506 rows, 14 columns — target: medv (median house value)

# 1. Visualise target
ggplot(Boston, aes(x = medv)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Boston House Values Distribution", x = "Median Value ($1,000)") +
  theme_minimal()

# 2. Train / test split (80 / 20)
set.seed(2024)
idx   <- sample(nrow(Boston), 0.8 * nrow(Boston))
train <- Boston[ idx, ]
test  <- Boston[-idx, ]

# 3. Fit on training data
model_bos <- lm(medv ~ ., data = train)
summary(model_bos)

# 4. Assumption checks
par(mfrow = c(2,2)); plot(model_bos); par(mfrow = c(1,1))
vif(model_bos)

# 5. Variable selection
model_bos_sel <- stepAIC(model_bos, direction = "both", trace = FALSE)
cat("Formula selected:\n"); print(formula(model_bos_sel))

# 6. Predict on test set
test_preds <- predict(model_bos_sel, newdata = test)

# 7. Quick evaluation (full metrics in 12_regression_evaluation.R)
rmse    <- sqrt(mean((test$medv - test_preds)^2))
mae     <- mean(abs(test$medv - test_preds))
r2_test <- 1 - sum((test$medv - test_preds)^2) / sum((test$medv - mean(train$medv))^2)

cat(sprintf("Test RMSE : %.3f\n", rmse))
cat(sprintf("Test MAE  : %.3f\n", mae))
cat(sprintf("Test R²   : %.3f\n", r2_test))

# 8. Actual vs Predicted plot
tibble(actual = test$medv, predicted = test_preds) |>
  ggplot(aes(x = predicted, y = actual)) +
  geom_point(alpha = 0.6, color = "steelblue", size = 2) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Boston Housing — Actual vs Predicted (Test Set)",
       x = "Predicted ($1,000)", y = "Actual ($1,000)") +
  theme_minimal()


# =============================================================================
#  QUICK-REFERENCE CARD
# =============================================================================
#
#  FIT:             lm(y ~ x1 + x2, data = df)
#  SUMMARY:         summary(model)
#  TIDY OUTPUT:     tidy(model, conf.int=TRUE)   glance(model)
#  PREDICTIONS:     predict(model, newdata, interval = "prediction")
#  DIAGNOSTICS:     plot(model)  |  autoplot(model)
#  INFLUENCE:       augment(model)  |  outlierTest(model)
#  VIF:             vif(model)
#  MODEL COMPARE:   anova(m1, m2)  |  AIC(m1, m2)
#  SELECTION:       stepAIC(model, direction = "both")
#
#  ASSUMPTION TESTS:
#    Linearity         plot(model, which=1)
#    Normality         shapiro.test(residuals(model))
#    Homoscedasticity  bptest(model)
#    Autocorrelation   dwtest(model)
#    Multicollinearity vif(model)
#
# =============================================================================
#  END OF 10_LINEAR_REGRESSION.R
# =============================================================================
