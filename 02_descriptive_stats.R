# =============================================================================
#  02_DESCRIPTIVE_STATS.R
#  Modelos de Predicción Analítica — Postgraduate Course
# =============================================================================
#
#  "Before you model, you must understand."
#
#  Descriptive statistics summarise and describe the main features of a
#  dataset. They are the MANDATORY first step before any predictive model:
#  they reveal data quality issues, guide feature engineering decisions,
#  and set realistic expectations for model performance.
#
#  PACKAGES NEEDED:
#    install.packages(c("tidyverse", "moments", "corrplot", "GGally",
#                       "skimr", "janitor"))
#
#  DATASET USED:
#    The built-in  mtcars  (motor trend cars) and
#    ggplot2::diamonds  (54,000 diamond prices)
#
#  SECTIONS:
#    1.  Measures of central tendency
#    2.  Measures of dispersion (spread)
#    3.  Shape: skewness and kurtosis
#    4.  Frequency tables and proportions
#    5.  The summary() family
#    6.  skimr::skim() — the analyst's best friend
#    7.  Detecting missing values
#    8.  Detecting outliers
#    9.  Correlation analysis
#    10. Grouped descriptive statistics
# =============================================================================

library(tidyverse)
library(moments)    # skewness(), kurtosis()
library(corrplot)   # visual correlation matrix
library(skimr)      # skim()
library(janitor)    # tabyl(), clean_names()


# ── Load datasets ─────────────────────────────────────────────────────────────
data(mtcars)
data(diamonds)   # from ggplot2, loaded with tidyverse

glimpse(mtcars)
glimpse(diamonds)


# =============================================================================
# SECTION 1 — MEASURES OF CENTRAL TENDENCY
# =============================================================================
#
#  These describe the "typical" value in a distribution.

x <- mtcars$mpg    # miles per gallon — our example variable

# Arithmetic mean — sensitive to outliers:
mean(x)

# Median — middle value, robust to outliers:
median(x)

# Mode — most frequent value (R has no built-in mode() for stats; write your own):
mode_val <- function(v) {
  tbl <- table(v)
  as.numeric(names(tbl)[which.max(tbl)])
}
mode_val(mtcars$cyl)   # most common number of cylinders

# Trimmed mean — drops the top and bottom 10% before computing:
mean(x, trim = 0.10)   # more robust than plain mean on skewed data

# Weighted mean — useful when observations have different importance:
# (example: weighting by engine size)
weighted.mean(x, w = mtcars$disp)

# WHEN TO USE WHICH:
#   Symmetric distribution        → mean ≈ median (either works)
#   Skewed distribution           → prefer median
#   Categorical / discrete data   → mode
#   Survey data with unequal groups → weighted mean


# =============================================================================
# SECTION 2 — MEASURES OF DISPERSION (SPREAD)
# =============================================================================
#
#  Dispersion tells you how spread out values are around the center.
#  A model trained on low-variance data learns very different things
#  from one trained on high-variance data.

# Range:
range(x)                       # min and max
diff(range(x))                 # span = max - min

# Variance — average squared deviation from the mean:
var(x)                         # R uses the unbiased formula (divides by n-1)

# Standard deviation — same units as the variable:
sd(x)

# Coefficient of Variation (CV) — relative spread, unit-free:
cv <- sd(x) / mean(x) * 100
cat("CV of mpg:", round(cv, 1), "%\n")
# CV < 15%: low variability; 15–30%: moderate; > 30%: high

# Interquartile Range (IQR) — range of the middle 50%, robust to outliers:
IQR(x)

# Quantiles:
quantile(x)                    # 0%, 25%, 50%, 75%, 100%
quantile(x, probs = c(0.1, 0.9))   # 10th and 90th percentile
quantile(x, probs = seq(0, 1, 0.2)) # quintiles

# Full five-number summary:
summary(x)    # Min, Q1, Median, Mean, Q3, Max


# =============================================================================
# SECTION 3 — SHAPE: SKEWNESS AND KURTOSIS
# =============================================================================
#
#  Many models (linear regression, naive Bayes, PCA) assume NORMALITY.
#  Skewness and kurtosis measure how far from normal a distribution is.

# ── SKEWNESS ──────────────────────────────────────────────────────────────────
#  Skewness = 0     : symmetric (normal-ish)
#  Skewness > 0     : right-skewed (long tail on the right, mean > median)
#  Skewness < 0     : left-skewed  (long tail on the left,  mean < median)
#  |Skewness| > 1   : substantially skewed — consider transforming

skewness(mtcars$mpg)    # slightly right-skewed
skewness(mtcars$hp)     # more skewed — high-HP outliers pull the tail

# Compare mean vs median to detect skew quickly:
mean(mtcars$hp) - median(mtcars$hp)   # positive = right-skewed

# ── KURTOSIS ──────────────────────────────────────────────────────────────────
#  Kurtosis (excess) = 0  : normal distribution
#  Kurtosis > 0           : leptokurtic — heavy tails, sharp peak
#  Kurtosis < 0           : platykurtic — light tails, flat peak

kurtosis(mtcars$mpg)    # excess kurtosis (already subtracts 3)

# ── COMMON TRANSFORMATIONS FOR SKEWED DATA ────────────────────────────────────
#  Right-skewed (positive): log, sqrt, 1/x
#  Left-skewed  (negative): x^2, x^3

hp_log  <- log(mtcars$hp)       # log transform
hp_sqrt <- sqrt(mtcars$hp)      # square root

skewness(mtcars$hp)             # before
skewness(hp_log)                # after log — much closer to 0
skewness(hp_sqrt)               # after sqrt — moderate improvement

# Shapiro-Wilk test for normality (works for n < 5,000):
shapiro.test(mtcars$mpg)    # p > 0.05 → cannot reject normality
shapiro.test(mtcars$hp)     # p < 0.05 → evidence of non-normality


# =============================================================================
# SECTION 4 — FREQUENCY TABLES AND PROPORTIONS
# =============================================================================

# For CATEGORICAL variables, count frequencies instead of computing means:

# Base R table():
table(mtcars$cyl)         # absolute frequencies
prop.table(table(mtcars$cyl))          # proportions
prop.table(table(mtcars$cyl)) * 100    # percentages

# Two-way (cross) table — cylinders × automatic/manual:
table(mtcars$cyl, mtcars$am)

# janitor::tabyl() — tidyverse-friendly, prettier output:
mtcars |> tabyl(cyl)
mtcars |> tabyl(cyl, am)              # cross-tabulation
mtcars |>
  tabyl(cyl, am) |>
  adorn_percentages("row") |>         # row percentages
  adorn_pct_formatting(digits = 1)    # format as %

# For diamonds dataset — cut quality distribution:
diamonds |> tabyl(cut) |> adorn_pct_formatting()

# count() from dplyr — best for pipelines:
diamonds |>
  count(cut, color) |>
  mutate(pct = n / sum(n) * 100) |>
  arrange(desc(n)) |>
  head(10)


# =============================================================================
# SECTION 5 — THE summary() FAMILY
# =============================================================================

# Base summary — the fastest first look:
summary(mtcars)

# summary() on a single column:
summary(mtcars$hp)

# dplyr summarise — compute exactly what you need:
mtcars |>
  summarise(across(where(is.numeric), list(
    mean   = \(x) round(mean(x, na.rm = TRUE), 2),
    sd     = \(x) round(sd(x, na.rm = TRUE), 2),
    median = \(x) median(x, na.rm = TRUE),
    min    = \(x) min(x, na.rm = TRUE),
    max    = \(x) max(x, na.rm = TRUE)
  )))
# across() applies a function to multiple columns at once
# where(is.numeric) selects only numeric columns


# =============================================================================
# SECTION 6 — skimr::skim() — THE ANALYST'S BEST FRIEND
# =============================================================================
#
#  skim() gives you a comprehensive data profile in one call:
#  n_missing, completion_rate, mean, sd, min/max, histograms, and more.
#  Use it at the START of every new project.

skim(mtcars)
skim(diamonds)

# Skim a subset of columns:
diamonds |>
  select(carat, price, depth, table) |>
  skim()

# Skim grouped by a category:
mtcars |>
  group_by(cyl) |>
  skim()


# =============================================================================
# SECTION 7 — DETECTING MISSING VALUES
# =============================================================================
#
#  Missing values (NA) silently break models. Always audit them first.

# Inject some NAs into a copy of mtcars for demonstration:
mtcars_na <- mtcars
set.seed(42)
mtcars_na[sample(1:nrow(mtcars_na), 5), "hp"]  <- NA
mtcars_na[sample(1:nrow(mtcars_na), 3), "mpg"] <- NA

# Total NAs in the dataset:
sum(is.na(mtcars_na))

# NAs per column:
colSums(is.na(mtcars_na))

# Proportion missing per column:
colMeans(is.na(mtcars_na)) * 100

# Tidy version — useful for filtering or plotting:
mtcars_na |>
  summarise(across(everything(),
                   list(n_miss = \(x) sum(is.na(x)),
                        pct_miss = \(x) round(mean(is.na(x)) * 100, 1)))) |>
  pivot_longer(everything(),
               names_to  = c("variable", ".value"),
               names_sep = "_(?=[^_]+$)")   # split on last underscore

# Which ROWS contain at least one NA:
mtcars_na[!complete.cases(mtcars_na), ]

# ── MISSINGNESS PATTERNS ──────────────────────────────────────────────────────
#  MCAR — Missing Completely At Random: safest; no bias
#  MAR  — Missing At Random: missing relates to OTHER observed variables
#  MNAR — Missing Not At Random: missing relates to the VALUE itself (worst)
#
#  We will cover imputation strategies in 05_missing_values.R


# =============================================================================
# SECTION 8 — DETECTING OUTLIERS
# =============================================================================
#
#  Outliers can dramatically distort regression coefficients, inflate RMSE,
#  and mislead feature importance rankings.
#  Always identify them before modelling — then decide: remove, cap, or keep.

# ── METHOD 1: IQR FENCE (Tukey) ───────────────────────────────────────────────
#  Lower fence = Q1 - 1.5 × IQR
#  Upper fence = Q3 + 1.5 × IQR
#  Values outside the fence are flagged as potential outliers.

detect_outliers_iqr <- function(x, mult = 1.5) {
  q1  <- quantile(x, 0.25, na.rm = TRUE)
  q3  <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  x < (q1 - mult * iqr) | x > (q3 + mult * iqr)
}

outlier_flags <- detect_outliers_iqr(mtcars$hp)
mtcars[outlier_flags, c("hp", "cyl", "disp")]   # outlier rows

# Apply to all numeric columns:
mtcars |>
  summarise(across(where(is.numeric),
                   \(x) sum(detect_outliers_iqr(x)))) |>
  pivot_longer(everything(), names_to = "variable", values_to = "n_outliers") |>
  arrange(desc(n_outliers))

# ── METHOD 2: Z-SCORE ─────────────────────────────────────────────────────────
#  Values more than 3 standard deviations from the mean are flagged.
#  Best for approximately normal distributions.

z_scores <- scale(mtcars$hp)     # standardise to mean=0, sd=1
outliers_z <- abs(z_scores) > 3
mtcars[outliers_z, "hp", drop = FALSE]   # none in this small dataset

# ── METHOD 3: VISUALISATION ───────────────────────────────────────────────────
#  Boxplot — outliers appear as individual points beyond the whiskers:
boxplot(mtcars$hp,
        main = "Horsepower Distribution",
        ylab = "HP",
        col  = "lightblue")

# Which values are in the outlier set:
boxplot.stats(mtcars$hp)$out


# =============================================================================
# SECTION 9 — CORRELATION ANALYSIS
# =============================================================================
#
#  Before building a model, understand how predictors relate to:
#   (a) the TARGET variable  → which features are most predictive?
#   (b) EACH OTHER           → multicollinearity can destabilise models

# Correlation of all numeric columns with the target (mpg):
cor(mtcars)[, "mpg"] |>
  sort(decreasing = TRUE) |>
  round(2)

# Full correlation matrix:
cor_matrix <- cor(mtcars, use = "complete.obs")   # handles NAs
round(cor_matrix, 2)

# Visualise with corrplot — much easier to read:
corrplot(cor_matrix,
         method  = "color",       # colour-coded squares
         type    = "upper",       # show upper triangle only
         addCoef.col = "black",   # print coefficients
         tl.cex  = 0.8,           # text label size
         number.cex = 0.7,
         title   = "mtcars Correlation Matrix",
         mar     = c(0, 0, 1, 0))

# Interpreting Pearson r:
#   |r| < 0.2  : negligible
#   |r| 0.2–0.4: weak
#   |r| 0.4–0.6: moderate
#   |r| 0.6–0.8: strong
#   |r| > 0.8  : very strong — possible multicollinearity!

# Multicollinearity warning — cyl, disp, and wt are all > 0.8 correlated:
cor_matrix[c("cyl", "disp", "wt"), c("cyl", "disp", "wt")] |> round(2)
# Including all three in the same model can cause instability.

# Correlation significance test:
cor.test(mtcars$mpg, mtcars$wt)   # r, t-statistic, p-value, CI


# =============================================================================
# SECTION 10 — GROUPED DESCRIPTIVE STATISTICS
# =============================================================================
#
#  Comparing distributions across subgroups is often the most informative
#  part of EDA and reveals patterns no overall summary can.

# Mean and SD of key variables by number of cylinders:
mtcars |>
  group_by(cyl) |>
  summarise(
    n           = n(),
    mpg_mean    = round(mean(mpg), 1),
    mpg_sd      = round(sd(mpg), 1),
    hp_mean     = round(mean(hp), 1),
    wt_mean     = round(mean(wt), 2),
    .groups = "drop"
  )

# Diamond price statistics by cut quality:
diamonds |>
  group_by(cut) |>
  summarise(
    n           = n(),
    price_mean  = round(mean(price)),
    price_median= median(price),
    price_sd    = round(sd(price)),
    pct_above_5k= round(mean(price > 5000) * 100, 1),
    .groups = "drop"
  ) |>
  arrange(desc(price_mean))

# ── ANOVA: ARE GROUP MEANS SIGNIFICANTLY DIFFERENT? ───────────────────────────
#  (covered in depth in the modelling module, but useful here for context)

aov_result <- aov(mpg ~ factor(cyl), data = mtcars)
summary(aov_result)
# Small p-value → cyl groups have significantly different mpg means → useful predictor


# =============================================================================
#  DESCRIPTIVE STATS CHECKLIST (use at the start of every project)
# =============================================================================
#
#   □ skim() or summary() — overall data profile
#   □ Dimensions (nrow, ncol), variable types
#   □ Missing values — which columns, what percentage
#   □ Numeric variables — mean, median, sd, skewness, outliers
#   □ Categorical variables — frequency tables, rare categories
#   □ Target variable — distribution, balance (if classification)
#   □ Correlations — with target + between predictors
#   □ Group comparisons — means by category
#   □ Time patterns — if a date column exists, plot over time
#
# =============================================================================
#  END OF 02_DESCRIPTIVE_STATS.R
#  Next: 03_visualization_ggplot2.R
# =============================================================================
