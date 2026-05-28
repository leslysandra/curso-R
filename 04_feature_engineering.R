# =============================================================================
#  04_FEATURE_ENGINEERING.R
#  Modelos de Predicción Analítica — Postgraduate Course
# =============================================================================
#
#  "Feature engineering is the art of turning raw data into information
#   a model can learn from. It is where domain expertise meets mathematics."
#
#  Feature engineering is often the HIGHEST-LEVERAGE activity in a machine
#  learning project — better features beat better algorithms. This script
#  covers every major technique you will use in this course.
#
#  PACKAGES NEEDED:
#    install.packages(c("tidyverse", "recipes", "caret", "lubridate",
#                       "fastDummies", "car"))
#
#  DATASETS USED:
#    ggplot2::diamonds, nycflights13::flights, mtcars
#
#  SECTIONS:
#    1.  What is feature engineering? (conceptual map)
#    2.  Handling missing values (imputation)
#    3.  Encoding categorical variables
#    4.  Numeric scaling and normalisation
#    5.  Variable transformations (skewness, log, Box-Cox)
#    6.  Creating new features from existing ones
#    7.  Date and time features
#    8.  Binning / discretisation
#    9.  Interaction terms
#    10. Dealing with multicollinearity (VIF)
#    11. The recipes package — a reproducible FE pipeline
# =============================================================================

library(tidyverse)
library(recipes)      # tidymodels-native feature engineering pipelines
library(lubridate)    # date/time manipulation
library(fastDummies)  # dummy_cols() for quick one-hot encoding
library(car)          # vif() for multicollinearity
library(nycflights13)

data(mtcars)
data(diamonds)


# =============================================================================
# SECTION 1 — WHAT IS FEATURE ENGINEERING? (conceptual map)
# =============================================================================
#
#  Raw Data (as collected)
#       │
#       ▼
#  ┌─────────────────────────────────────────────────────────────┐
#  │                  FEATURE ENGINEERING                        │
#  │                                                             │
#  │  Impute missing  →  Encode categories  →  Scale numerics   │
#  │       │                    │                    │           │
#  │  Transform skew  →  Create new features  →  Remove redund. │
#  └─────────────────────────────────────────────────────────────┘
#       │
#       ▼
#  Model-Ready Feature Matrix  (X)  +  Target Vector  (y)
#
#  RULES TO REMEMBER:
#   1. FE is fit on TRAINING data only — never peek at test data.
#   2. The same transformations must be applied identically at prediction time.
#   3. Document every step — reproducibility is non-negotiable.
#   4. Domain knowledge beats blind automation; ask "why would this help?"


# =============================================================================
# SECTION 2 — HANDLING MISSING VALUES (IMPUTATION)
# =============================================================================
#
#  Most models cannot handle NA. You must decide: remove rows, remove
#  columns, or IMPUTE (fill in estimated values).
#
#  NEVER impute with test data statistics. Fit imputation on training set,
#  apply to test set.

# Create a dataset with artificial missing values:
set.seed(42)
diamonds_na <- diamonds |>
  mutate(
    price = if_else(runif(n()) < 0.05, NA_real_, as.numeric(price)),
    depth = if_else(runif(n()) < 0.08, NA_real_, depth),
    cut   = if_else(runif(n()) < 0.03, NA_character_, as.character(cut))
  )

cat("Missing values per column:\n")
colSums(is.na(diamonds_na))

# ── STRATEGY 1: Remove rows with ANY missing value (listwise deletion) ─────────
#  Only acceptable if missingness is < 5% and MCAR.
diamonds_complete <- diamonds_na |> drop_na()
nrow(diamonds_complete)   # rows lost

# ── STRATEGY 2: Remove COLUMNS that are mostly missing ────────────────────────
pct_missing <- colMeans(is.na(diamonds_na)) * 100
cols_to_keep <- names(pct_missing[pct_missing < 30])  # keep < 30% missing
diamonds_filtered <- diamonds_na |> select(all_of(cols_to_keep))

# ── STRATEGY 3: Mean / Median imputation (numeric) ────────────────────────────
#  Simple, fast. Mean for symmetric; median for skewed distributions.

impute_median <- function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  x
}

diamonds_imputed <- diamonds_na |>
  mutate(
    price = impute_median(price),
    depth = impute_median(depth)
  )

# ── STRATEGY 4: Mode imputation (categorical) ─────────────────────────────────
mode_val <- function(x) {
  tbl <- table(x, useNA = "no")
  names(tbl)[which.max(tbl)]
}

diamonds_imputed <- diamonds_imputed |>
  mutate(cut = if_else(is.na(cut), mode_val(cut), cut))

# ── STRATEGY 5: Group-wise imputation ─────────────────────────────────────────
#  Impute price within each cut group (smarter than overall median)

diamonds_group_imputed <- diamonds_na |>
  group_by(cut) |>
  mutate(price = if_else(is.na(price),
                         median(price, na.rm = TRUE),
                         as.numeric(price))) |>
  ungroup()

# ── STRATEGY 6: Constant / indicator imputation ───────────────────────────────
#  Fill with a sentinel value AND add a binary "was_missing" indicator.
#  The indicator lets the model learn whether missingness itself is predictive.

diamonds_indicator <- diamonds_na |>
  mutate(
    depth_missing = as.integer(is.na(depth)),   # 1 = was missing
    depth         = if_else(is.na(depth), -999, depth)  # sentinel
  )

# NOTE: For advanced imputation (KNN, MICE, random forest) see the recipes
# package demonstrated in Section 11.


# =============================================================================
# SECTION 3 — ENCODING CATEGORICAL VARIABLES
# =============================================================================
#
#  Most models require NUMERIC inputs. Categorical columns must be converted.

# ── METHOD 1: Integer (ordinal) encoding ──────────────────────────────────────
#  Use ONLY when the category has a meaningful natural order.

cut_levels <- c("Fair", "Good", "Very Good", "Premium", "Ideal")
diamonds <- diamonds |>
  mutate(cut_ordinal = as.integer(factor(cut, levels = cut_levels)))

diamonds |> count(cut, cut_ordinal) |> arrange(cut_ordinal)

# ── METHOD 2: One-Hot Encoding (OHE) / Dummy variables ───────────────────────
#  Creates one binary (0/1) column per category.
#  IMPORTANT: drop one column (the reference level) to avoid the
#  "dummy variable trap" — perfect multicollinearity.

# Base R approach:
dummy_matrix <- model.matrix(~ cut - 1, data = diamonds)   # -1 removes intercept
head(dummy_matrix)

# fastDummies approach — easier with data frames:
diamonds_dummies <- diamonds |>
  select(carat, cut, price) |>
  dummy_cols("cut",
             remove_first_dummy  = TRUE,    # drop reference category
             remove_selected_columns = TRUE) # remove original column
head(diamonds_dummies)

# ── METHOD 3: Target Encoding (mean encoding) ─────────────────────────────────
#  Replace each category with the MEAN of the target variable for that group.
#  Very powerful for high-cardinality variables (many categories).
#  CRITICAL: compute means on TRAINING data only to avoid leakage.

# Example: encode diamond "color" with its mean price
color_means <- diamonds |>
  group_by(color) |>
  summarise(color_target_enc = mean(price), .groups = "drop")

diamonds_target_enc <- diamonds |>
  left_join(color_means, by = "color")

diamonds_target_enc |> select(color, price, color_target_enc) |> head(10)

# ── METHOD 4: Frequency Encoding ─────────────────────────────────────────────
#  Replace category with how often it appears. Useful for counts / rank.

color_freq <- diamonds |>
  count(color) |>
  rename(color_freq = n)

diamonds_freq_enc <- diamonds |>
  left_join(color_freq, by = "color")

# ── CHOOSING ENCODING METHOD ──────────────────────────────────────────────────
#  Ordinal, naturally ordered   → integer encoding
#  Nominal, low cardinality (< 10 levels) → one-hot encoding
#  Nominal, high cardinality (≥ 10 levels) → target or frequency encoding
#  Tree-based models (RF, XGBoost) → can handle integer encoding natively


# =============================================================================
# SECTION 4 — NUMERIC SCALING AND NORMALISATION
# =============================================================================
#
#  WHY SCALE?
#   - Distance-based models (KNN, SVM, k-means) are sensitive to scale.
#   - Regularised models (Ridge, Lasso) penalise coefficients — unscaled
#     variables are penalised unequally.
#   - Gradient-based methods (neural networks) converge faster when scaled.
#   - Tree models (Random Forest, XGBoost) are SCALE-INVARIANT — no need.

# Original ranges:
summary(mtcars[, c("mpg", "disp", "hp", "wt")])
# mpg: 10–34; disp: 71–472; hp: 52–335; wt: 1.5–5.4

# ── METHOD 1: Standardisation (Z-score) ───────────────────────────────────────
#  Subtracts mean, divides by SD → mean = 0, SD = 1
#  Best for: linear regression, logistic regression, SVM, PCA, neural nets

scale_zscore <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

mtcars_scaled <- mtcars |>
  mutate(across(c(mpg, disp, hp, wt), scale_zscore, .names = "{.col}_z"))

summary(mtcars_scaled |> select(ends_with("_z")))  # all ~ mean=0, sd=1

# base R shortcut (returns a matrix):
scaled_matrix <- scale(mtcars[, c("mpg", "disp", "hp", "wt")])
attr(scaled_matrix, "scaled:center")   # means used (save these for test set!)
attr(scaled_matrix, "scaled:scale")    # SDs used

# ── METHOD 2: Min-Max Normalisation ───────────────────────────────────────────
#  Rescales to [0, 1]. Sensitive to outliers.
#  Best for: neural networks, image data

scale_minmax <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

mtcars_mm <- mtcars |>
  mutate(across(c(mpg, disp, hp, wt), scale_minmax, .names = "{.col}_mm"))

summary(mtcars_mm |> select(ends_with("_mm")))   # all in [0, 1]

# ── METHOD 3: Robust Scaling (Median / IQR) ───────────────────────────────────
#  Uses median and IQR instead of mean and SD → outlier-robust

scale_robust <- function(x) {
  (x - median(x, na.rm = TRUE)) / IQR(x, na.rm = TRUE)
}

mtcars_robust <- mtcars |>
  mutate(across(c(mpg, disp, hp, wt), scale_robust, .names = "{.col}_r"))

# !! CRITICAL WORKFLOW NOTE !!
# Compute scaling parameters (mean, SD, min, max) on TRAINING DATA ONLY.
# Apply the SAME parameters to the test set.
# Never re-fit scaling on test data — that is data leakage.


# =============================================================================
# SECTION 5 — VARIABLE TRANSFORMATIONS
# =============================================================================
#
#  Many models assume (or perform better with) symmetric, near-normal
#  distributions. Transformations reduce skewness and stabilise variance.

# Visualise skewness in diamonds:
hist(diamonds$price, breaks = 50, main = "Price — raw (right-skewed)")
hist(log(diamonds$price), breaks = 50, main = "Price — log (more symmetric)")

# ── LOG TRANSFORMATION: most common for right-skewed positive variables ────────
diamonds <- diamonds |>
  mutate(log_price = log(price),      # natural log
         log1p_carat = log1p(carat))  # log(1+x) — safe when x can be 0

# ── SQUARE ROOT: gentler than log, also for right-skewed data ─────────────────
diamonds <- diamonds |>
  mutate(sqrt_price = sqrt(price))

# ── BOX-COX TRANSFORMATION: finds the optimal power λ automatically ───────────
#  λ =  1  → no change
#  λ =  0  → log transformation
#  λ = 0.5 → square root
#  λ = -1  → reciprocal

library(MASS)   # already available in base R installation
bc <- MASS::boxcox(price ~ 1, data = diamonds |> slice_sample(n = 1000))
lambda_optimal <- bc$x[which.max(bc$y)]
cat("Optimal Box-Cox lambda for price:", round(lambda_optimal, 3), "\n")

# Apply Box-Cox manually:
boxcox_transform <- function(x, lambda) {
  if (abs(lambda) < 1e-10) log(x) else (x^lambda - 1) / lambda
}

diamonds <- diamonds |>
  mutate(bc_price = boxcox_transform(price, lambda_optimal))

# ── YEO-JOHNSON: Like Box-Cox but handles ZERO and NEGATIVE values ────────────
# (available via the recipes package — see Section 11)

# Compare skewness before/after:
library(moments)
cat("Price skewness:         ", round(skewness(diamonds$price),     3), "\n")
cat("log(Price) skewness:    ", round(skewness(diamonds$log_price),  3), "\n")
cat("sqrt(Price) skewness:   ", round(skewness(diamonds$sqrt_price), 3), "\n")
cat("Box-Cox(Price) skewness:", round(skewness(diamonds$bc_price),   3), "\n")


# =============================================================================
# SECTION 6 — CREATING NEW FEATURES FROM EXISTING ONES
# =============================================================================
#
#  The most creative part of FE — domain knowledge is king here.

# ── Ratio features ────────────────────────────────────────────────────────────
mtcars <- mtcars |>
  mutate(
    hp_per_wt   = hp / wt,              # power-to-weight ratio
    mpg_per_cyl = mpg / as.numeric(cyl) # efficiency per cylinder
  )

# ── Polynomial features ───────────────────────────────────────────────────────
#  Capture non-linear relationships in linear models.
#  Use sparingly — risk of overfitting with too many polynomials.

mtcars <- mtcars |>
  mutate(
    wt_sq  = wt^2,   # quadratic term
    wt_cub = wt^3    # cubic term (rarely needed)
  )

# Check: is the relationship between wt and mpg better with wt²?
lm_linear <- lm(mpg ~ wt, data = mtcars)
lm_quad   <- lm(mpg ~ wt + wt_sq, data = mtcars)

summary(lm_linear)$r.squared    # R² without wt²
summary(lm_quad)$r.squared      # R² with    wt² — typically improves

# ── Aggregate features (entity-level summary statistics) ──────────────────────
#  Summarise related rows into a single observation-level feature.

carrier_stats <- flights |>
  group_by(carrier) |>
  summarise(
    carrier_mean_delay = mean(dep_delay, na.rm = TRUE),
    carrier_pct_delay  = mean(dep_delay > 15, na.rm = TRUE),
    carrier_n_flights  = n(),
    .groups = "drop"
  )

# Join back to the flight-level data:
flights_enriched <- flights |>
  left_join(carrier_stats, by = "carrier")

# Now each flight row has its carrier's historical delay profile as features.

# ── Flag / binary indicator features ─────────────────────────────────────────
flights_enriched <- flights_enriched |>
  mutate(
    is_weekend     = wday(time_hour, week_start = 1) %in% c(6, 7),
    is_morning     = hour %in% 5:11,
    is_long_haul   = distance > 1500,
    is_jfk         = origin == "JFK"
  )

# ── Difference features ───────────────────────────────────────────────────────
#  Capture the GAP between related variables.

flights_enriched <- flights_enriched |>
  mutate(
    delay_difference = dep_delay - arr_delay   # did the crew make up time?
  )


# =============================================================================
# SECTION 7 — DATE AND TIME FEATURES
# =============================================================================
#
#  Raw datetime columns are useless to most models.
#  Decompose them into numeric and categorical components.

flights_dates <- flights |>
  select(time_hour, dep_delay) |>
  filter(!is.na(dep_delay)) |>
  mutate(
    # Decomposition:
    year        = year(time_hour),
    month       = month(time_hour),                        # 1–12
    month_name  = month(time_hour, label = TRUE),          # Jan, Feb…
    day_of_month= mday(time_hour),                         # 1–31
    day_of_week = wday(time_hour, label = TRUE, week_start = 1), # Mon–Sun
    week_of_year= week(time_hour),                         # 1–53
    quarter     = quarter(time_hour),                      # 1–4
    hour_of_day = hour(time_hour),                         # 0–23

    # Derived flags:
    is_weekend  = wday(time_hour, week_start = 1) %in% c(6, 7),
    is_peak_hour= hour(time_hour) %in% c(7, 8, 17, 18, 19),
    is_holiday_month = month(time_hour) %in% c(7, 11, 12),

    # Cyclical encoding — preserves the circular nature of time:
    # (hour 0 and hour 23 are adjacent, not 23 units apart)
    hour_sin    = sin(2 * pi * hour(time_hour) / 24),
    hour_cos    = cos(2 * pi * hour(time_hour) / 24),
    month_sin   = sin(2 * pi * month(time_hour) / 12),
    month_cos   = cos(2 * pi * month(time_hour) / 12)
  )

head(flights_dates)

# Average delay by hour — shows the pattern cyclical encoding captures:
flights_dates |>
  group_by(hour_of_day) |>
  summarise(mean_delay = mean(dep_delay), .groups = "drop") |>
  ggplot(aes(x = hour_of_day, y = mean_delay)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(size = 2) +
  labs(title = "Average Departure Delay by Hour of Day",
       x = "Hour", y = "Mean Delay (min)") +
  theme_minimal()


# =============================================================================
# SECTION 8 — BINNING / DISCRETISATION
# =============================================================================
#
#  Converting a continuous variable into ordered categories.
#  Useful when: the relationship is non-linear and step-like,
#  or when you want to group rare extreme values.

# ── Equal-width bins (cut) ────────────────────────────────────────────────────
mtcars <- mtcars |>
  mutate(
    hp_band = cut(hp, breaks = 3,
                  labels = c("Low", "Mid", "High"))
  )
table(mtcars$hp_band)

# ── Equal-frequency bins / quantile bins (cut + quantile) ─────────────────────
#  Each bin contains roughly the same number of observations.
mtcars <- mtcars |>
  mutate(
    hp_quartile = cut(hp,
                      breaks = quantile(hp, probs = c(0, 0.25, 0.5, 0.75, 1)),
                      labels = c("Q1", "Q2", "Q3", "Q4"),
                      include.lowest = TRUE)
  )
table(mtcars$hp_quartile)

# ── Custom domain-driven bins ─────────────────────────────────────────────────
diamonds <- diamonds |>
  mutate(
    carat_size = case_when(
      carat < 0.5  ~ "Small",
      carat < 1.0  ~ "Medium",
      carat < 2.0  ~ "Large",
      TRUE         ~ "Very Large"
    ),
    carat_size = factor(carat_size,
                        levels = c("Small", "Medium", "Large", "Very Large"))
  )
table(diamonds$carat_size)

# CAUTION: Binning always loses information. Only use it when the
# non-linear, step-wise pattern is supported by domain knowledge or EDA.


# =============================================================================
# SECTION 9 — INTERACTION TERMS
# =============================================================================
#
#  An interaction feature captures the COMBINED effect of two predictors
#  that is not explained by either alone.
#  e.g., "heavy AND low-power" cars → worse mpg than either factor alone suggests.

# Create interaction manually:
mtcars <- mtcars |>
  mutate(
    wt_hp_interaction = wt * hp   # numeric × numeric
  )

# In linear models, R's formula interface handles interactions directly:
lm_interaction <- lm(mpg ~ wt * hp, data = mtcars)  # wt + hp + wt:hp
summary(lm_interaction)

# wt:cyl — numeric × categorical interaction:
lm_cat_int <- lm(mpg ~ wt * cyl, data = mtcars)    # slope of wt differs by cyl
summary(lm_cat_int)

# NOTE:
#  *  in a formula → main effects + interaction   (wt * hp = wt + hp + wt:hp)
#  :  in a formula → interaction ONLY             (wt:hp)


# =============================================================================
# SECTION 10 — DEALING WITH MULTICOLLINEARITY (VIF)
# =============================================================================
#
#  Multicollinearity: two or more predictors are highly correlated.
#  Effect on models: inflated standard errors, unstable coefficients,
#  unreliable p-values, poor interpretability.
#  Does NOT hurt tree models — it only affects linear/logistic regression.
#
#  Variance Inflation Factor (VIF):
#    VIF = 1       : no correlation
#    VIF = 1–5     : mild — acceptable
#    VIF = 5–10    : moderate — investigate
#    VIF > 10      : severe — remove or combine one of the correlated variables

model_full <- lm(mpg ~ disp + hp + wt + cyl, data = mtcars |>
                   mutate(cyl = as.numeric(cyl)))

vif(model_full)   # from the car package

# disp, hp, wt, and cyl are all highly correlated → high VIF expected.

# Solutions for high VIF:
#   1. Remove one of the correlated variables (keep the more interpretable one)
#   2. Combine them into a new feature (e.g., hp / wt → power-to-weight)
#   3. Apply Principal Component Analysis (PCA) to decorrelate predictors
#   4. Use Ridge regression (handles multicollinearity via regularisation)

# Reduced model — keep only wt and hp:
model_reduced <- lm(mpg ~ wt + hp, data = mtcars)
vif(model_reduced)   # now much lower


# =============================================================================
# SECTION 11 — THE recipes PACKAGE: REPRODUCIBLE FE PIPELINES
# =============================================================================
#
#  recipes lets you define a preprocessing BLUEPRINT that:
#   - Is fit ONLY on training data
#   - Can be applied identically to test data and new observations
#   - Is completely reproducible and shareable
#
#  WORKFLOW:
#    recipe()   → define steps (blueprint)
#    prep()     → fit the blueprint on training data
#    bake()     → apply the fitted blueprint to any dataset

# ── Split data first ──────────────────────────────────────────────────────────
set.seed(42)
n     <- nrow(diamonds)
idx   <- sample(seq_len(n), size = 0.8 * n)
train <- diamonds[idx, ]
test  <- diamonds[-idx, ]

# ── Define the recipe ─────────────────────────────────────────────────────────
diamond_recipe <- recipe(price ~ ., data = train) |>

  # Step 1: Remove ID or near-zero variance columns (if any):
  step_zv(all_predictors()) |>           # remove zero-variance columns
  step_nzv(all_predictors()) |>          # remove near-zero variance

  # Step 2: Impute missing values:
  step_impute_median(all_numeric_predictors()) |>     # median for numerics
  step_impute_mode(all_nominal_predictors()) |>       # mode for categoricals

  # Step 3: Log-transform the target and skewed predictors:
  step_log(price, carat, base = exp(1), skip = FALSE) |>  # natural log

  # Step 4: Create polynomial feature for carat:
  step_poly(carat, degree = 2) |>

  # Step 5: One-hot encode categorical variables:
  step_dummy(all_nominal_predictors(), one_hot = FALSE) |>  # drops one level

  # Step 6: Standardise all numeric predictors:
  step_normalize(all_numeric_predictors())

# ── Inspect the recipe ────────────────────────────────────────────────────────
diamond_recipe   # prints the steps

# ── Prep: FIT the recipe on training data ─────────────────────────────────────
diamond_prep <- prep(diamond_recipe, training = train)
diamond_prep   # now shows computed parameters (means, SDs, etc.)

# ── Bake: APPLY the fitted recipe ─────────────────────────────────────────────
train_processed <- bake(diamond_prep, new_data = NULL)   # NULL → returns training set
test_processed  <- bake(diamond_prep, new_data = test)   # apply same recipe to test

glimpse(train_processed)
dim(train_processed)
dim(test_processed)

# All preprocessing decisions (means, SDs, mode values, dummy levels) were
# learned from  train  ONLY and applied identically to  test.
# This is the correct, leakage-free workflow.

# ── Common recipe steps quick reference ───────────────────────────────────────
#
#  IMPUTATION:
#   step_impute_mean()    step_impute_median()   step_impute_mode()
#   step_impute_knn()     step_impute_bag()      (model-based)
#
#  ENCODING:
#   step_dummy()           one-hot (drops one)
#   step_other()           lump rare categories into "Other"
#   step_unknown()         make NA its own level
#   step_integer()         ordinal integer encoding
#
#  NUMERIC TRANSFORMS:
#   step_log()             log transform
#   step_sqrt()            square root
#   step_BoxCox()          Box-Cox (positive values)
#   step_YeoJohnson()      Yeo-Johnson (any sign)
#   step_normalize()       standardise (Z-score)
#   step_range()           min-max scale to [0,1]
#   step_robust()          median/IQR scaling
#
#  FEATURE CREATION:
#   step_poly()            polynomial terms
#   step_interact()        interaction terms
#   step_ratio()           ratios between columns
#   step_date()            decompose Date column into components
#
#  SELECTION / REDUCTION:
#   step_zv()              remove zero-variance
#   step_corr()            remove highly correlated predictors (VIF mitigation)
#   step_pca()             principal component analysis
#   step_select()          keep/drop specific columns


# =============================================================================
#  FEATURE ENGINEERING CHECKLIST
# =============================================================================
#
#   □ Audit missing values — strategy per column
#   □ Identify categorical columns — choose encoding method
#   □ Check distribution of each numeric predictor — log/sqrt if skewed
#   □ Check distribution of TARGET variable — transform if needed
#   □ Create domain-driven features (ratios, flags, aggregates)
#   □ Decompose datetime columns into numeric components
#   □ Add cyclical encoding for periodic variables (hour, month)
#   □ Add polynomial terms if EDA shows non-linear relationship
#   □ Check VIF — remove or combine highly correlated predictors
#   □ Scale numeric features (if using distance- or penalty-based model)
#   □ Wrap everything in a recipe() — fit on train, apply to test
#   □ Verify: train and test have the SAME columns and types after bake()
#
# =============================================================================
#  END OF 04_FEATURE_ENGINEERING.R
#  Next: 05_train_test_split.R  →  then model building begins!
# =============================================================================
