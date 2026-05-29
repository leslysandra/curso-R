# =============================================================================
#  03_VISUALIZATION_GGPLOT2.R
#  Modelos de Predicción Analítica
# =============================================================================
#
#  "A picture is worth a thousand regression coefficients."
#
#  ggplot2 is the gold standard for data visualisation in R.
#  It is built on the Grammar of Graphics: every chart is assembled from
#  composable layers, giving you complete control over every visual element.
#
#  PACKAGES NEEDED:
#    install.packages(c("tidyverse", "scales", "patchwork", "ggcorrplot",
#                       "GGally", "ggridges", "viridis"))
#
#  DATASETS USED:
#    mtcars, diamonds (ggplot2), nycflights13::flights, gapminder
#
#  SECTIONS:
#    1.  The Grammar of Graphics — how ggplot2 thinks
#    2.  Histograms and density plots    (one numeric variable)
#    3.  Box plots and violin plots      (numeric × categorical)
#    4.  Scatter plots                   (two numeric variables)
#    5.  Bar charts                      (categorical variable)
#    6.  Line charts                     (trends over time)
#    7.  Correlation heatmaps
#    8.  Faceting — small multiples
#    9.  Themes and styling
#    10. Combining plots with patchwork
#    11. Plots specifically for predictive modelling
# =============================================================================

library(tidyverse)
library(scales)       # number formatting on axes
library(patchwork)    # combine multiple ggplots
library(ggcorrplot)   # ggplot2-style correlation matrix
library(GGally)       # ggpairs — scatterplot matrix
library(ggridges)     # ridge / joy plots
library(viridis)      # colour-blind friendly palettes

data(mtcars)
data(diamonds)

# Convert some columns to factors for nicer plotting:
mtcars <- mtcars |>
  mutate(cyl = factor(cyl),
         am  = factor(am, labels = c("Automatic", "Manual")),
         vs  = factor(vs, labels = c("V-shaped", "Straight")))


# =============================================================================
# SECTION 1 — THE GRAMMAR OF GRAPHICS: HOW ggplot2 THINKS
# =============================================================================
#
#  Every ggplot2 chart is built from LAYERS stacked with  +
#
#  ggplot(data, aes(...))   ← the canvas: which data, which aesthetics
#    + geom_*()             ← the geometry: what shape to draw
#    + stat_*()             ← optional: statistical transformation
#    + scale_*()            ← control axes, colours, sizes
#    + coord_*()            ← coordinate system
#    + facet_*()            ← small multiples
#    + theme_*() + theme()  ← all non-data visual elements
#
#  AESTHETIC MAPPINGS (inside aes()):
#    x, y         → position on axes
#    color/colour → outline or line colour
#    fill         → filled area colour
#    size         → size of points / lines
#    shape        → point shape (circle, triangle…)
#    alpha        → transparency (0 = invisible, 1 = opaque)
#    linetype     → solid, dashed, dotted…
#    group        → grouping without visual change
#
#  FIXED vs MAPPED aesthetics:
#    Inside  aes()  → mapped to a VARIABLE (varies per row)
#    Outside aes()  → fixed for the entire layer
#
#  Examples:
#    geom_point(aes(color = cyl))    ← colour maps to the cyl column
#    geom_point(color = "steelblue") ← colour is fixed for all points

# Minimal working example:
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point()

# Add a layer at a time — this is the key mental model:
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(aes(color = cyl), size = 3) +
  geom_smooth(method = "lm", color = "black", se = TRUE) +
  labs(title = "Weight vs Fuel Efficiency",
       x = "Weight (1000 lbs)", y = "Miles per Gallon",
       color = "Cylinders")


# =============================================================================
# SECTION 2 — HISTOGRAMS AND DENSITY PLOTS (one numeric variable)
# =============================================================================
#
#  Use these to understand the DISTRIBUTION of a variable.
#  Always do this for your target variable AND for every numeric predictor.

# Histogram — count of observations in each bin:
ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Fuel Efficiency",
       x = "Miles per Gallon", y = "Count")

# Adjust bins to see different resolutions:
ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(binwidth = 2, fill = "steelblue", color = "white") +
  labs(title = "MPG — binwidth = 2")

# Density plot — smooth estimate of the probability distribution:
ggplot(mtcars, aes(x = mpg)) +
  geom_density(fill = "steelblue", alpha = 0.5) +
  labs(title = "MPG Density Curve")

# Overlapping densities by group — great for comparing distributions:
ggplot(mtcars, aes(x = mpg, fill = cyl)) +
  geom_density(alpha = 0.4) +
  labs(title = "MPG Distribution by Number of Cylinders",
       x = "Miles per Gallon", fill = "Cylinders")

# Histogram + density curve together:
ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(aes(y = after_stat(density)),   # normalise y to density
                 bins = 10, fill = "steelblue", color = "white") +
  geom_density(color = "red", linewidth = 1) +
  labs(title = "MPG Histogram with Density Overlay")

# Ridge plots — compare distributions across MANY groups elegantly:
ggplot(diamonds, aes(x = price, y = cut, fill = cut)) +
  geom_density_ridges(alpha = 0.7, scale = 1.2) +
  scale_x_continuous(labels = dollar) +
  scale_fill_viridis_d() +
  labs(title = "Diamond Price Distribution by Cut Quality",
       x = "Price (USD)", y = "Cut") +
  theme_minimal() +
  theme(legend.position = "none")


# =============================================================================
# SECTION 3 — BOX PLOTS AND VIOLIN PLOTS (numeric × categorical)
# =============================================================================
#
#  Box plots show median, IQR, and outliers.
#  Violin plots add the full distribution shape.

# Box plot:
ggplot(mtcars, aes(x = cyl, y = mpg, fill = cyl)) +
  geom_boxplot(alpha = 0.7, outlier.color = "red", outlier.size = 2) +
  labs(title = "Fuel Efficiency by Number of Cylinders",
       x = "Cylinders", y = "MPG") +
  theme_minimal() +
  theme(legend.position = "none")

# Violin plot — shows distribution shape (wider = more data):
ggplot(mtcars, aes(x = cyl, y = mpg, fill = cyl)) +
  geom_violin(alpha = 0.6, trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
  labs(title = "MPG Distribution by Cylinders (Violin + Box)",
       x = "Cylinders", y = "MPG") +
  theme_minimal() +
  theme(legend.position = "none")

# Add individual data points for small datasets:
ggplot(mtcars, aes(x = cyl, y = mpg, fill = cyl)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_jitter(width = 0.1, size = 2, alpha = 0.7) +  # jitter avoids overlap
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "MPG by Cylinders — Violin + Jittered Points")

# Grouped box plot — two categorical variables:
ggplot(mtcars, aes(x = cyl, y = mpg, fill = am)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "MPG by Cylinders and Transmission Type",
       x = "Cylinders", y = "MPG", fill = "Transmission") +
  theme_minimal()


# =============================================================================
# SECTION 4 — SCATTER PLOTS (two numeric variables)
# =============================================================================
#
#  Essential for exploring RELATIONSHIPS between variables.
#  This is the most-used chart in predictive analytics EDA.

# Basic scatter:
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 3, color = "steelblue", alpha = 0.8) +
  labs(title = "Weight vs Fuel Efficiency", x = "Weight (klb)", y = "MPG")

# Add a linear trend line (regression line):
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +  # se = confidence band
  labs(title = "Linear Trend: Weight → MPG")

# Add a non-linear (LOESS) trend:
ggplot(diamonds |> sample_n(2000), aes(x = carat, y = price)) +
  geom_point(alpha = 0.2, size = 1) +
  geom_smooth(method = "loess", color = "red", linewidth = 1.2) +
  scale_y_continuous(labels = dollar) +
  labs(title = "Carat vs Price (LOESS smooth, n=2000)")

# Map colour and size to additional variables:
ggplot(mtcars, aes(x = wt, y = mpg, color = cyl, size = hp)) +
  geom_point(alpha = 0.8) +
  scale_size_continuous(range = c(2, 8)) +
  labs(title = "Weight vs MPG — colour=cylinders, size=horsepower",
       color = "Cylinders", size = "Horsepower")

# Annotate specific points:
ggplot(mtcars, aes(x = wt, y = mpg, label = rownames(mtcars))) +
  geom_point(size = 2, color = "steelblue") +
  geom_text(hjust = -0.1, size = 2.5, check_overlap = TRUE) +
  labs(title = "Labelled Scatter — mtcars")

# 2D density for large datasets (instead of overplotted scatter):
ggplot(diamonds, aes(x = carat, y = price)) +
  geom_bin2d(bins = 60) +
  scale_fill_viridis_c() +
  scale_y_continuous(labels = dollar) +
  labs(title = "2D Bin Density — Carat vs Price (54,000 diamonds)",
       fill = "Count")


# =============================================================================
# SECTION 5 — BAR CHARTS (categorical variables)
# =============================================================================

# Frequency bar chart (geom_bar counts for you):
ggplot(diamonds, aes(x = cut, fill = cut)) +
  geom_bar() +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "Diamond Count by Cut Quality", x = "Cut", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# Proportional / percent bar chart:
ggplot(diamonds, aes(x = cut, fill = cut)) +
  geom_bar(aes(y = after_stat(count / sum(count)))) +
  scale_y_continuous(labels = percent) +
  labs(title = "Proportion of Diamonds by Cut", y = "Percentage") +
  theme_minimal() +
  theme(legend.position = "none")

# Stacked bar — two categorical variables:
ggplot(diamonds, aes(x = cut, fill = color)) +
  geom_bar(position = "stack") +
  labs(title = "Diamond Count: Cut × Colour (stacked)")

# 100% stacked — compare proportions regardless of group size:
ggplot(diamonds, aes(x = cut, fill = color)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  labs(title = "Diamond Colour Mix within Each Cut (100% stacked)", y = "%")

# Horizontal bar chart — easier to read with long category names:
ggplot(diamonds, aes(x = fct_rev(fct_infreq(cut)), fill = cut)) +
  geom_bar() +
  coord_flip() +
  labs(title = "Cut Frequency (sorted)", x = "Cut", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# Column chart with computed values (geom_col — you supply the height):
avg_price_by_cut <- diamonds |>
  group_by(cut) |>
  summarise(mean_price = mean(price), .groups = "drop")

ggplot(avg_price_by_cut, aes(x = reorder(cut, mean_price), y = mean_price, fill = cut)) +
  geom_col() +
  geom_text(aes(label = dollar(round(mean_price))),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_y_continuous(labels = dollar, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Average Diamond Price by Cut", x = "Cut", y = "Mean Price") +
  theme_minimal() +
  theme(legend.position = "none")


# =============================================================================
# SECTION 6 — LINE CHARTS (trends over time)
# =============================================================================

# Average flight delay per month (using nycflights13):
# install.packages("nycflights13")
library(nycflights13)

monthly_delay <- flights |>
  filter(!is.na(dep_delay)) |>
  group_by(month) |>
  summarise(mean_delay = mean(dep_delay), .groups = "drop")

ggplot(monthly_delay, aes(x = month, y = mean_delay)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(color = "steelblue", size = 3) +
  scale_x_continuous(breaks = 1:12,
                     labels = month.abb) +   # Jan, Feb, …
  labs(title = "Average Departure Delay by Month — NYC Flights 2013",
       x = NULL, y = "Mean Delay (minutes)") +
  theme_minimal()

# Multiple lines — one per carrier (top 5 by volume):
top5 <- flights |> count(carrier, sort = TRUE) |> slice_head(n = 5) |> pull(carrier)

carrier_delay <- flights |>
  filter(!is.na(dep_delay), carrier %in% top5) |>
  group_by(carrier, month) |>
  summarise(mean_delay = mean(dep_delay), .groups = "drop") |>
  left_join(airlines, by = "carrier")

ggplot(carrier_delay, aes(x = month, y = mean_delay, color = name)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = 1:12, labels = month.abb) +
  labs(title = "Monthly Delay Trends — Top 5 Airlines",
       x = NULL, y = "Mean Delay (min)", color = "Airline") +
  theme_minimal() +
  theme(legend.position = "bottom")


# =============================================================================
# SECTION 7 — CORRELATION HEATMAPS
# =============================================================================

# Compute correlation matrix:
cor_mat <- cor(select(mtcars |> mutate(cyl = as.numeric(cyl),
                                       am  = as.numeric(am),
                                       vs  = as.numeric(vs)),
                      where(is.numeric)),
               use = "complete.obs")

# ggcorrplot — ggplot2-native, more customisable than corrplot:
ggcorrplot(cor_mat,
           method     = "square",
           type       = "lower",
           lab        = TRUE,         # show coefficients
           lab_size   = 3,
           colors     = c("#d73027", "white", "#1a9850"),  # red–white–green
           title      = "mtcars Correlation Matrix",
           ggtheme    = theme_minimal())

# GGally::ggpairs — scatterplot matrix: distributions + correlations + scatter:
mtcars |>
  select(mpg, wt, hp, disp) |>
  ggpairs(title = "Scatterplot Matrix — mtcars key variables")
# This single call replaces dozens of individual plots.
# Diagonal = distribution; upper triangle = correlation value; lower = scatter.


# =============================================================================
# SECTION 8 — FACETING: SMALL MULTIPLES
# =============================================================================
#
#  Faceting repeats the same chart across subgroups — one of the most
#  powerful tools for comparing patterns across categories.

# facet_wrap — wraps panels in a grid:
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(color = "steelblue", size = 2) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  facet_wrap(~ cyl, labeller = label_both) +
  labs(title = "Weight vs MPG — Faceted by Cylinders")

# facet_grid — two-dimensional grid of panels:
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  facet_grid(vs ~ cyl, labeller = label_both) +
  labs(title = "Weight vs MPG — Rows = Engine Shape, Cols = Cylinders")

# Faceted density for diamonds:
ggplot(diamonds, aes(x = price, fill = cut)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ cut, ncol = 1) +
  scale_x_continuous(labels = dollar) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Price Distribution by Cut") +
  theme_minimal() +
  theme(legend.position = "none")


# =============================================================================
# SECTION 9 — THEMES AND STYLING
# =============================================================================

base_plot <- ggplot(mtcars, aes(x = wt, y = mpg, color = cyl)) +
  geom_point(size = 3) +
  labs(title = "Weight vs MPG", x = "Weight (klb)", y = "MPG", color = "Cyl")

base_plot + theme_grey()        # default
base_plot + theme_bw()          # black & white — good for print
base_plot + theme_minimal()     # clean, no background grid lines
base_plot + theme_classic()     # classic axes, no grid
base_plot + theme_dark()        # dark background
base_plot + theme_light()       # light grey panel

# Fine-grained customisation with theme():
base_plot +
  theme_minimal() +
  theme(
    plot.title   = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title   = element_text(size = 12),
    axis.text    = element_text(size = 10),
    legend.position = "bottom",
    panel.grid.minor = element_blank()   # remove minor grid lines
  )

# Colour palettes:
base_plot + scale_color_brewer(palette = "Set1")      # qualitative
base_plot + scale_color_viridis_d()                   # colour-blind safe
base_plot + scale_color_manual(values = c("4" = "#e41a1c",
                                          "6" = "#377eb8",
                                          "8" = "#4daf4a"))

# Saving a plot to disk (adjust width/height in inches, dpi for resolution):
# ggsave("plots/weight_vs_mpg.png", plot = last_plot(),
#         width = 8, height = 5, dpi = 300)


# =============================================================================
# SECTION 10 — COMBINING PLOTS WITH patchwork
# =============================================================================

p1 <- ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "white") +
  labs(title = "MPG Distribution") + theme_minimal()

p2 <- ggplot(mtcars, aes(x = cyl, y = mpg, fill = cyl)) +
  geom_boxplot(alpha = 0.7) +
  theme_minimal() + theme(legend.position = "none") +
  labs(title = "MPG by Cylinders")

p3 <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(aes(color = cyl), size = 3) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  theme_minimal() +
  labs(title = "Weight vs MPG")

p4 <- ggplot(mtcars, aes(x = hp, y = mpg)) +
  geom_point(aes(color = cyl), size = 3) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  theme_minimal() +
  labs(title = "Horsepower vs MPG")

# Arrange plots:
(p1 | p2) / (p3 | p4)   # top row: p1 + p2; bottom row: p3 + p4

# Add a shared title:
(p1 | p2) / (p3 | p4) +
  plot_annotation(
    title    = "Exploratory Analysis — mtcars Dataset",
    subtitle = "Fuel Efficiency (MPG) examined from multiple angles",
    theme    = theme(plot.title = element_text(size = 16, face = "bold"))
  )


# =============================================================================
# SECTION 11 — PLOTS SPECIFICALLY FOR PREDICTIVE MODELLING
# =============================================================================

# ── 11a. Actual vs Predicted plot (residual diagnostics) ─────────────────────
model <- lm(mpg ~ wt + hp + cyl, data = mtcars)

mtcars_pred <- mtcars |>
  mutate(
    predicted = predict(model),
    residual  = mpg - predicted
  )

# Actual vs Predicted — ideal: points on the diagonal line:
ggplot(mtcars_pred, aes(x = predicted, y = mpg)) +
  geom_point(color = "steelblue", size = 3, alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted MPG",
       x = "Predicted", y = "Actual") +
  theme_minimal()

# ── 11b. Residual plot — check linearity and homoscedasticity ─────────────────
ggplot(mtcars_pred, aes(x = predicted, y = residual)) +
  geom_point(color = "steelblue", size = 3, alpha = 0.8) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "orange") +
  labs(title = "Residuals vs Fitted Values",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal()
# Ideal: points randomly scattered around 0 with constant spread.

# ── 11c. Variable Importance plot ────────────────────────────────────────────
# (Using a simple standardised coefficient as proxy for importance)
importance_df <- data.frame(
  variable   = c("wt", "hp", "cyl8", "cyl6"),
  importance = c(0.82, 0.44, 0.38, 0.21)
)

ggplot(importance_df, aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance (example)",
       x = "Predictor", y = "Relative Importance") +
  theme_minimal()

# ── 11d. Distribution of target variable — before and after transform ─────────
p_before <- ggplot(diamonds, aes(x = price)) +
  geom_histogram(bins = 50, fill = "tomato", color = "white") +
  labs(title = "Price — Original (right-skewed)",
       x = "Price (USD)") +
  theme_minimal()

p_after <- ggplot(diamonds, aes(x = log(price))) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Price — Log Transformed (more symmetric)",
       x = "log(Price)") +
  theme_minimal()

p_before | p_after   # compare side by side

# ── 11e. Correlation with target — dot plot ───────────────────────────────────
cor_with_mpg <- cor(
  mtcars |> mutate(cyl = as.numeric(cyl), am = as.numeric(am), vs = as.numeric(vs)),
  use = "complete.obs"
)[, "mpg"] |>
  as.data.frame() |>
  setNames("correlation") |>
  rownames_to_column("variable") |>
  filter(variable != "mpg") |>
  arrange(correlation)

ggplot(cor_with_mpg, aes(x = correlation,
                          y = reorder(variable, correlation),
                          fill = correlation > 0)) +
  geom_col() +
  geom_vline(xintercept = 0, linewidth = 0.8) +
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato"),
                    labels = c("Negative", "Positive")) +
  labs(title = "Correlation of Each Variable with MPG",
       x = "Pearson r", y = NULL, fill = "Direction") +
  theme_minimal()


# =============================================================================
#  ggplot2 QUICK REFERENCE
# =============================================================================
#
#  GEOMS (chart types):
#   geom_point()     scatter plot
#   geom_line()      line chart
#   geom_bar()       bar chart (counts rows)
#   geom_col()       bar chart (uses your y value)
#   geom_histogram() histogram
#   geom_density()   density curve
#   geom_boxplot()   box plot
#   geom_violin()    violin plot
#   geom_smooth()    trend line
#   geom_hline() / geom_vline()  horizontal/vertical reference lines
#   geom_text() / geom_label()   text annotations
#   geom_tile()      heatmap
#
#  SCALES:
#   scale_x/y_continuous()   numeric axis
#   scale_x/y_log10()        log scale axis
#   scale_color/fill_brewer() categorical palettes
#   scale_color/fill_viridis_c/d()  continuous/discrete viridis
#   scale_x/y_continuous(labels = dollar / percent / comma)
#
#  COORDINATES:
#   coord_flip()     swap x and y
#   coord_fixed()    equal aspect ratio
#   coord_polar()    pie / donut (avoid for data science!)
#
# =============================================================================
#  END OF 03_VISUALIZATION_GGPLOT2.R
#  Next: 04_feature_engineering.R
# =============================================================================
